use super::*;

#[cube(launch_unchecked)]
fn rope_rotate_pairs_kernel(
    input: &Array<f32>,
    cos: &Array<f32>,
    sin: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *head_dim;
    if row >= *rows {
        terminate!();
    }
    let dim = idx % *head_dim;
    let pair = dim / 2;
    let parity = dim % 2;
    let token = (row / *heads) % *tokens;
    let base = row * *head_dim + pair * 2;
    let even = input[base];
    let odd = input[base + 1];
    let trig_idx = token * *pairs + pair;
    let cos_v = cos[trig_idx];
    let sin_v = sin[trig_idx];
    let rotated_even = even * cos_v - odd * sin_v;
    let rotated_odd = even * sin_v + odd * cos_v;
    if parity == 0 {
        output[idx] = rotated_even;
    } else {
        output[idx] = rotated_odd;
    }
}

#[cube(launch_unchecked)]
fn rope_rotate_pairs_phase_kernel(
    input: &Array<f32>,
    phase: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *head_dim;
    if row >= *rows {
        terminate!();
    }
    let dim = idx % *head_dim;
    let pair = dim / 2;
    let parity = dim % 2;
    let token = (row / *heads) % *tokens;
    let base = row * *head_dim + pair * 2;
    let even = input[base];
    let odd = input[base + 1];
    let trig_idx = token * *pairs + pair;
    let phase_v = phase[trig_idx];
    let cos_v = phase_v.cos();
    let sin_v = phase_v.sin();
    let rotated_even = even * cos_v - odd * sin_v;
    let rotated_odd = even * sin_v + odd * cos_v;
    if parity == 0 {
        output[idx] = rotated_even;
    } else {
        output[idx] = rotated_odd;
    }
}

#[cube(launch_unchecked)]
fn rope_rotate_pairs_coords_kernel(
    input: &Array<f32>,
    coords: &Array<i32>,
    pair_freq: &Array<f32>,
    pair_axis: &Array<i32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *head_dim;
    if row >= *rows {
        terminate!();
    }
    let dim = idx % *head_dim;
    let pair = dim / 2;
    let parity = dim % 2;
    let token = (row / *heads) % *tokens;
    let base = row * *head_dim + pair * 2;
    let even = input[base];
    let odd = input[base + 1];

    let axis = pair_axis[pair];
    let mut cos_v = 1.0f32;
    let mut sin_v = 0.0f32;
    if axis >= 0 {
        let coord_base = token * 3;
        let coord = if axis == 0 {
            coords[coord_base]
        } else if axis == 1 {
            coords[coord_base + 1]
        } else {
            coords[coord_base + 2]
        };
        let phase_v = f32::cast_from(coord) * pair_freq[pair];
        cos_v = phase_v.cos();
        sin_v = phase_v.sin();
    }
    let rotated_even = even * cos_v - odd * sin_v;
    let rotated_odd = even * sin_v + odd * cos_v;
    if parity == 0 {
        output[idx] = rotated_even;
    } else {
        output[idx] = rotated_odd;
    }
}

#[cube(launch_unchecked)]
fn linear_skinny_kernel(
    input: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    in_channels: &usize,
    out_channels: &usize,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *out_channels;
    if row >= *rows {
        terminate!();
    }
    let out_idx = idx % *out_channels;
    let input_base = row * *in_channels;
    let weight_base = out_idx * *in_channels;
    let mut acc = bias[out_idx];
    for in_idx in 0..*in_channels {
        acc += input[input_base + in_idx] * weight[weight_base + in_idx];
    }
    output[idx] = acc;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_kernel(
    input: &Array<f32>,
    stats: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let base = row * *channels;
    let mut sum = 0.0f32;
    for channel in 0..*channels {
        sum += input[base + channel];
    }
    let mean = sum / f32::cast_from(*channels);
    let mut sq_sum = 0.0f32;
    for channel in 0..*channels {
        let centered = input[base + channel] - mean;
        sq_sum += centered * centered;
    }
    let var = sq_sum / f32::cast_from(*channels);
    let stats_base = row * 2;
    stats[stats_base] = mean;
    stats[stats_base + 1] = var;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_partial_kernel(
    input: &Array<f32>,
    partials: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
    chunk_size: &usize,
) {
    if ABSOLUTE_POS >= *rows * *chunks {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *chunks;
    if row >= *rows {
        terminate!();
    }
    let chunk = idx % *chunks;
    let start = chunk * *chunk_size;
    let input_base = row * *channels;
    let mut sum = 0.0f32;
    for offset in 0..*chunk_size {
        let channel = start + offset;
        if channel < *channels {
            let value = input[input_base + channel];
            sum += value;
        }
    }
    partials[idx] = sum;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_f16_kernel(
    input: &Array<f16>,
    stats: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let base = row * *channels;
    let mut sum = 0.0f32;
    for channel in 0..*channels {
        sum += f32::cast_from(input[base + channel]);
    }
    let mean = sum / f32::cast_from(*channels);
    let mut sq_sum = 0.0f32;
    for channel in 0..*channels {
        let centered = f32::cast_from(input[base + channel]) - mean;
        sq_sum += centered * centered;
    }
    let var = sq_sum / f32::cast_from(*channels);
    let stats_base = row * 2;
    stats[stats_base] = mean;
    stats[stats_base + 1] = var;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_partial_f16_kernel(
    input: &Array<f16>,
    partials: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
    chunk_size: &usize,
) {
    if ABSOLUTE_POS >= *rows * *chunks {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *chunks;
    if row >= *rows {
        terminate!();
    }
    let chunk = idx % *chunks;
    let start = chunk * *chunk_size;
    let input_base = row * *channels;
    let mut sum = 0.0f32;
    for offset in 0..*chunk_size {
        let channel = start + offset;
        if channel < *channels {
            sum += f32::cast_from(input[input_base + channel]);
        }
    }
    partials[idx] = sum;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_reduce_mean_kernel(
    partials: &Array<f32>,
    stats: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let mut sum = 0.0f32;
    let partial_base = row * *chunks;
    for chunk in 0..*chunks {
        sum += partials[partial_base + chunk];
    }
    let channels_f = f32::cast_from(*channels);
    let mean = sum / channels_f;
    let stats_base = row * 2;
    stats[stats_base] = mean;
    stats[stats_base + 1] = 0.0;
}

#[cube(launch_unchecked)]
fn layer_norm_row_var_partial_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    partials: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
    chunk_size: &usize,
) {
    if ABSOLUTE_POS >= *rows * *chunks {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *chunks;
    if row >= *rows {
        terminate!();
    }
    let chunk = idx % *chunks;
    let start = chunk * *chunk_size;
    let input_base = row * *channels;
    let mean = stats[row * 2];
    let mut sq_sum = 0.0f32;
    for offset in 0..*chunk_size {
        let channel = start + offset;
        if channel < *channels {
            let centered = input[input_base + channel] - mean;
            sq_sum += centered * centered;
        }
    }
    partials[idx] = sq_sum;
}

#[cube(launch_unchecked)]
fn layer_norm_row_var_partial_f16_kernel(
    input: &Array<f16>,
    stats: &Array<f32>,
    partials: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
    chunk_size: &usize,
) {
    if ABSOLUTE_POS >= *rows * *chunks {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *chunks;
    if row >= *rows {
        terminate!();
    }
    let chunk = idx % *chunks;
    let start = chunk * *chunk_size;
    let input_base = row * *channels;
    let mean = stats[row * 2];
    let mut sq_sum = 0.0f32;
    for offset in 0..*chunk_size {
        let channel = start + offset;
        if channel < *channels {
            let centered = f32::cast_from(input[input_base + channel]) - mean;
            sq_sum += centered * centered;
        }
    }
    partials[idx] = sq_sum;
}

#[cube(launch_unchecked)]
fn layer_norm_row_stats_reduce_var_kernel(
    partials: &Array<f32>,
    stats: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    chunks: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let mut sq_sum = 0.0f32;
    let partial_base = row * *chunks;
    for chunk in 0..*chunks {
        sq_sum += partials[partial_base + chunk];
    }
    let var = sq_sum / f32::cast_from(*channels);
    stats[row * 2 + 1] = var;
}

#[cube(launch_unchecked)]
fn layer_norm_affine_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *channels;
    if row >= *rows {
        terminate!();
    }
    let channel = idx % *channels;
    let stats_base = row * 2;
    let mean = stats[stats_base];
    let var = stats[stats_base + 1];
    let inv_std = (var + *eps).sqrt().recip();
    let centered = input[idx] - mean;
    output[idx] = centered * inv_std * weight[channel] + bias[channel];
}

#[cube(launch_unchecked)]
fn layer_norm_affine_silu_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    weight: &Array<f32>,
    bias: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    channels: &usize,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *channels;
    if row >= *rows {
        terminate!();
    }
    let channel = idx % *channels;
    let stats_base = row * 2;
    let mean = stats[stats_base];
    let var = stats[stats_base + 1];
    let inv_std = (var + *eps).sqrt().recip();
    let centered = input[idx] - mean;
    let affine = centered * inv_std * weight[channel] + bias[channel];
    let silu = affine / (1.0 + (-affine).exp());
    output[idx] = silu;
}

#[cube(launch_unchecked)]
fn layer_norm_modulated_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    scale: &Array<f32>,
    shift: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    channels: &usize,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *channels;
    if row >= *rows {
        terminate!();
    }
    let channel = idx % *channels;
    let batch = row / *tokens;
    let mod_idx = batch * *channels + channel;
    let stats_base = row * 2;
    let mean = stats[stats_base];
    let var = stats[stats_base + 1];
    let inv_std = (var + *eps).sqrt().recip();
    let centered = input[idx] - mean;
    output[idx] = centered * inv_std * (scale[mod_idx] + 1.0) + shift[mod_idx];
}

#[cube(launch_unchecked)]
fn layer_norm_modulated_f16_kernel(
    input: &Array<f16>,
    stats: &Array<f32>,
    scale: &Array<f16>,
    shift: &Array<f16>,
    output: &mut Array<f16>,
    rows: &usize,
    tokens: &usize,
    channels: &usize,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *channels;
    if row >= *rows {
        terminate!();
    }
    let channel = idx % *channels;
    let batch = row / *tokens;
    let mod_idx = batch * *channels + channel;
    let stats_base = row * 2;
    let mean = stats[stats_base];
    let var = stats[stats_base + 1];
    let inv_std = (var + *eps).sqrt().recip();
    let centered = f32::cast_from(input[idx]) - mean;
    let scale_v = f32::cast_from(scale[mod_idx]);
    let shift_v = f32::cast_from(shift[mod_idx]);
    output[idx] = f16::cast_from(centered * inv_std * (scale_v + 1.0) + shift_v);
}

fn launch_layer_norm_row_stats(
    input: &CubeTensor<burn_wgpu::WgpuRuntime>,
    stats: &CubeTensor<burn_wgpu::WgpuRuntime>,
    rows: usize,
    channels: usize,
    cube_dim: CubeDim,
) -> Result<(), String> {
    if layer_norm_partial_stats_enabled(rows, channels) {
        let chunks = channels.div_ceil(LAYER_NORM_STATS_PARTIAL_CHUNK);
        let partial_elements = rows
            .checked_mul(chunks)
            .ok_or_else(|| "layer norm partial stats size overflow".to_string())?;
        let partial_bytes = partial_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "layer norm partial stats byte size overflow".to_string())?;
        let partials = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::new([partial_elements]),
            input.client.empty(partial_bytes),
            DType::F32,
        );
        let partial_work = rows
            .checked_mul(chunks)
            .ok_or_else(|| "layer norm partial stats work size overflow".to_string())?;
        let partial_cube_count =
            calculate_cube_count_elemwise(&input.client, partial_work, cube_dim);
        unsafe {
            layer_norm_row_stats_partial_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                partial_cube_count.clone(),
                cube_dim,
                input.clone().into_array_arg(),
                partials.clone().into_array_arg(),
                rows,
                channels,
                chunks,
                LAYER_NORM_STATS_PARTIAL_CHUNK,
            )
            .map_err(|err| format!("layer_norm_row_stats_partial_kernel launch failed: {err:?}"))?;
        }
        let row_cube_count = calculate_cube_count_elemwise(&input.client, rows, cube_dim);
        unsafe {
            layer_norm_row_stats_reduce_mean_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                row_cube_count.clone(),
                cube_dim,
                partials.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                rows,
                channels,
                chunks,
            )
            .map_err(|err| {
                format!("layer_norm_row_stats_reduce_mean_kernel launch failed: {err:?}")
            })?;
        }
        unsafe {
            layer_norm_row_var_partial_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                partial_cube_count,
                cube_dim,
                input.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                partials.clone().into_array_arg(),
                rows,
                channels,
                chunks,
                LAYER_NORM_STATS_PARTIAL_CHUNK,
            )
            .map_err(|err| format!("layer_norm_row_var_partial_kernel launch failed: {err:?}"))?;
        }
        unsafe {
            layer_norm_row_stats_reduce_var_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                row_cube_count,
                cube_dim,
                partials.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                rows,
                channels,
                chunks,
            )
            .map_err(|err| {
                format!("layer_norm_row_stats_reduce_var_kernel launch failed: {err:?}")
            })?;
        }
        return Ok(());
    }

    let row_cube_count = calculate_cube_count_elemwise(&input.client, rows, cube_dim);
    unsafe {
        layer_norm_row_stats_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input.client,
            row_cube_count,
            cube_dim,
            input.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            rows,
            channels,
        )
        .map_err(|err| format!("layer_norm_row_stats_kernel launch failed: {err:?}"))?;
    }
    Ok(())
}

fn launch_layer_norm_row_stats_f16(
    input: &CubeTensor<burn_wgpu::WgpuRuntime>,
    stats: &CubeTensor<burn_wgpu::WgpuRuntime>,
    rows: usize,
    channels: usize,
    cube_dim: CubeDim,
) -> Result<(), String> {
    if layer_norm_partial_stats_enabled(rows, channels) {
        let chunks = channels.div_ceil(LAYER_NORM_STATS_PARTIAL_CHUNK);
        let partial_elements = rows
            .checked_mul(chunks)
            .ok_or_else(|| "layer norm f16 partial stats size overflow".to_string())?;
        let partial_bytes = partial_elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or_else(|| "layer norm f16 partial stats byte size overflow".to_string())?;
        let partials = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::new([partial_elements]),
            input.client.empty(partial_bytes),
            DType::F32,
        );
        let partial_work = rows
            .checked_mul(chunks)
            .ok_or_else(|| "layer norm f16 partial stats work size overflow".to_string())?;
        let partial_cube_count =
            calculate_cube_count_elemwise(&input.client, partial_work, cube_dim);
        unsafe {
            layer_norm_row_stats_partial_f16_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                partial_cube_count.clone(),
                cube_dim,
                input.clone().into_array_arg(),
                partials.clone().into_array_arg(),
                rows,
                channels,
                chunks,
                LAYER_NORM_STATS_PARTIAL_CHUNK,
            )
            .map_err(|err| {
                format!("layer_norm_row_stats_partial_f16_kernel launch failed: {err:?}")
            })?;
        }
        let row_cube_count = calculate_cube_count_elemwise(&input.client, rows, cube_dim);
        unsafe {
            layer_norm_row_stats_reduce_mean_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                row_cube_count.clone(),
                cube_dim,
                partials.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                rows,
                channels,
                chunks,
            )
            .map_err(|err| {
                format!("layer_norm_row_stats_reduce_mean_kernel launch failed: {err:?}")
            })?;
        }
        unsafe {
            layer_norm_row_var_partial_f16_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                partial_cube_count,
                cube_dim,
                input.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                partials.clone().into_array_arg(),
                rows,
                channels,
                chunks,
                LAYER_NORM_STATS_PARTIAL_CHUNK,
            )
            .map_err(|err| {
                format!("layer_norm_row_var_partial_f16_kernel launch failed: {err:?}")
            })?;
        }
        unsafe {
            layer_norm_row_stats_reduce_var_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                &input.client,
                row_cube_count,
                cube_dim,
                partials.clone().into_array_arg(),
                stats.clone().into_array_arg(),
                rows,
                channels,
                chunks,
            )
            .map_err(|err| {
                format!("layer_norm_row_stats_reduce_var_kernel launch failed: {err:?}")
            })?;
        }
        return Ok(());
    }

    let row_cube_count = calculate_cube_count_elemwise(&input.client, rows, cube_dim);
    unsafe {
        layer_norm_row_stats_f16_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input.client,
            row_cube_count,
            cube_dim,
            input.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            rows,
            channels,
        )
        .map_err(|err| format!("layer_norm_row_stats_f16_kernel launch failed: {err:?}"))?;
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn layer_norm_row_stats_debug_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 2>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let [rows, channels] = input.dims();
    let stats_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "layer norm debug stats size overflow".to_string())?;
    let stats_bytes = stats_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm debug stats byte size overflow".to_string())?;
    let input_p = input.into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, 2]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    launch_layer_norm_row_stats(&input_p, &stats, rows, channels, resolve_cube_dim())?;
    Ok(BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(
        TensorPrimitive::Float(stats),
    ))
}

#[cube(launch_unchecked)]
fn multihead_rms_norm_rope_coords_row_affine_kernel(
    input: &Array<f32>,
    gamma: &Array<f32>,
    coords: &Array<i32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
    rope_freq_0: &f32,
    rope_freq_1_ln: &f32,
    scale: &f32,
    eps: &f32,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let base = row * *head_dim;
    let head = row % *heads;
    let token = (row / *heads) % *tokens;
    let gamma_base = head * *head_dim;

    let mut sq_sum = 0.0f32;
    for channel in 0..*head_dim {
        let value = input[base + channel];
        sq_sum += value * value;
    }
    let inv_rms = (sq_sum + *eps).sqrt().recip();

    for pair in 0..*pairs {
        let pair_channel = pair * 2;
        let even_idx = base + pair_channel;
        let odd_idx = even_idx + 1;
        let even = input[even_idx] * inv_rms * *scale * gamma[gamma_base + pair_channel];
        let odd = input[odd_idx] * inv_rms * *scale * gamma[gamma_base + pair_channel + 1];

        let mut freq_dim = *pairs / 3;
        if freq_dim < 1 {
            freq_dim = 1;
        }
        let coord_base = token * 3;
        let mut phase = 0.0f32;
        if pair < freq_dim {
            let exp = f32::cast_from(pair) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base]) * freq;
        } else if pair < freq_dim * 2 {
            let freq_idx = pair - freq_dim;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 1]) * freq;
        } else if pair < freq_dim * 3 {
            let freq_idx = pair - freq_dim * 2;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 2]) * freq;
        }
        let c = phase.cos();
        let s = phase.sin();
        output[even_idx] = even * c - odd * s;
        output[odd_idx] = even * s + odd * c;
    }
}

#[cube(launch_unchecked)]
fn multihead_qk_rms_norm_rope_coords_from_qkv_kernel(
    input: &Array<f32>,
    q_gamma: &Array<f32>,
    k_gamma: &Array<f32>,
    coords: &Array<i32>,
    q_output: &mut Array<f32>,
    k_output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
    rope_freq_0: &f32,
    rope_freq_1_ln: &f32,
    scale: &f32,
    eps: &f32,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let base = row * *head_dim;
    let head = row % *heads;
    let token = (row / *heads) % *tokens;
    let batch = row / (*tokens * *heads);
    let qkv_batch_token_base = ((batch * *tokens + token) * 3 * *heads) * *head_dim;
    let q_input_base = qkv_batch_token_base + head * *head_dim;
    let k_input_base = qkv_batch_token_base + (*heads + head) * *head_dim;
    let gamma_base = head * *head_dim;

    let mut q_sq_sum = 0.0f32;
    let mut k_sq_sum = 0.0f32;
    for channel in 0..*head_dim {
        let q_value = input[q_input_base + channel];
        let k_value = input[k_input_base + channel];
        q_sq_sum += q_value * q_value;
        k_sq_sum += k_value * k_value;
    }
    let q_inv_rms = (q_sq_sum + *eps).sqrt().recip();
    let k_inv_rms = (k_sq_sum + *eps).sqrt().recip();

    for pair in 0..*pairs {
        let pair_channel = pair * 2;
        let q_even_idx = q_input_base + pair_channel;
        let q_odd_idx = q_even_idx + 1;
        let k_even_idx = k_input_base + pair_channel;
        let k_odd_idx = k_even_idx + 1;
        let out_even_idx = base + pair_channel;
        let out_odd_idx = out_even_idx + 1;

        let q_even = input[q_even_idx] * q_inv_rms * *scale * q_gamma[gamma_base + pair_channel];
        let q_odd = input[q_odd_idx] * q_inv_rms * *scale * q_gamma[gamma_base + pair_channel + 1];
        let k_even = input[k_even_idx] * k_inv_rms * *scale * k_gamma[gamma_base + pair_channel];
        let k_odd = input[k_odd_idx] * k_inv_rms * *scale * k_gamma[gamma_base + pair_channel + 1];

        let mut freq_dim = *pairs / 3;
        if freq_dim < 1 {
            freq_dim = 1;
        }
        let coord_base = token * 3;
        let mut phase = 0.0f32;
        if pair < freq_dim {
            let exp = f32::cast_from(pair) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base]) * freq;
        } else if pair < freq_dim * 2 {
            let freq_idx = pair - freq_dim;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 1]) * freq;
        } else if pair < freq_dim * 3 {
            let freq_idx = pair - freq_dim * 2;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 2]) * freq;
        }
        let c = phase.cos();
        let s = phase.sin();
        q_output[out_even_idx] = q_even * c - q_odd * s;
        q_output[out_odd_idx] = q_even * s + q_odd * c;
        k_output[out_even_idx] = k_even * c - k_odd * s;
        k_output[out_odd_idx] = k_even * s + k_odd * c;
    }
}

#[cube(launch_unchecked)]
fn multihead_qkv_module_rms_norm_rope_coords_from_qkv_kernel(
    input: &Array<f32>,
    q_gamma: &Array<f32>,
    k_gamma: &Array<f32>,
    coords: &Array<i32>,
    q_output: &mut Array<f32>,
    k_output: &mut Array<f32>,
    v_output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
    rope_freq_0: &f32,
    rope_freq_1_ln: &f32,
    scale: &f32,
    eps: &f32,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let head = row % *heads;
    let token = (row / *heads) % *tokens;
    let batch = row / (*tokens * *heads);
    let qkv_batch_token_base = ((batch * *tokens + token) * 3 * *heads) * *head_dim;
    let q_input_base = qkv_batch_token_base + head * *head_dim;
    let k_input_base = qkv_batch_token_base + (*heads + head) * *head_dim;
    let v_input_base = qkv_batch_token_base + ((*heads * 2) + head) * *head_dim;
    let module_base = ((batch * *heads + head) * *tokens + token) * *head_dim;
    let gamma_base = head * *head_dim;

    let mut q_sq_sum = 0.0f32;
    let mut k_sq_sum = 0.0f32;
    for channel in 0..*head_dim {
        let q_value = input[q_input_base + channel];
        let k_value = input[k_input_base + channel];
        q_sq_sum += q_value * q_value;
        k_sq_sum += k_value * k_value;
        v_output[module_base + channel] = input[v_input_base + channel];
    }
    let q_inv_rms = (q_sq_sum + *eps).sqrt().recip();
    let k_inv_rms = (k_sq_sum + *eps).sqrt().recip();

    for pair in 0..*pairs {
        let pair_channel = pair * 2;
        let q_even_idx = q_input_base + pair_channel;
        let q_odd_idx = q_even_idx + 1;
        let k_even_idx = k_input_base + pair_channel;
        let k_odd_idx = k_even_idx + 1;
        let out_even_idx = module_base + pair_channel;
        let out_odd_idx = out_even_idx + 1;

        let q_even = input[q_even_idx] * q_inv_rms * *scale * q_gamma[gamma_base + pair_channel];
        let q_odd = input[q_odd_idx] * q_inv_rms * *scale * q_gamma[gamma_base + pair_channel + 1];
        let k_even = input[k_even_idx] * k_inv_rms * *scale * k_gamma[gamma_base + pair_channel];
        let k_odd = input[k_odd_idx] * k_inv_rms * *scale * k_gamma[gamma_base + pair_channel + 1];

        let mut freq_dim = *pairs / 3;
        if freq_dim < 1 {
            freq_dim = 1;
        }
        let coord_base = token * 3;
        let mut phase = 0.0f32;
        if pair < freq_dim {
            let exp = f32::cast_from(pair) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base]) * freq;
        } else if pair < freq_dim * 2 {
            let freq_idx = pair - freq_dim;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 1]) * freq;
        } else if pair < freq_dim * 3 {
            let freq_idx = pair - freq_dim * 2;
            let exp = f32::cast_from(freq_idx) / f32::cast_from(freq_dim);
            let freq = *rope_freq_0 * (-*rope_freq_1_ln * exp).exp();
            phase = f32::cast_from(coords[coord_base + 2]) * freq;
        }
        let c = phase.cos();
        let s = phase.sin();
        q_output[out_even_idx] = q_even * c - q_odd * s;
        q_output[out_odd_idx] = q_even * s + q_odd * c;
        k_output[out_even_idx] = k_even * c - k_odd * s;
        k_output[out_odd_idx] = k_even * s + k_odd * c;
    }
}

#[cube(launch_unchecked)]
fn multihead_rms_norm_row_stats_kernel(
    input: &Array<f32>,
    stats: &mut Array<f32>,
    rows: &usize,
    head_dim: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let base = row * *head_dim;
    let mut sq_sum = 0.0f32;
    for channel in 0..*head_dim {
        let value = input[base + channel];
        sq_sum += value * value;
    }
    stats[row] = sq_sum;
}

#[cube(launch_unchecked)]
fn multihead_rms_norm_affine_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    gamma: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    heads: &usize,
    head_dim: &usize,
    scale: &f32,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let idx = ABSOLUTE_POS;
    let row = idx / *head_dim;
    if row >= *rows {
        terminate!();
    }
    let channel = idx % *head_dim;
    let head = row % *heads;
    let gamma_idx = head * *head_dim + channel;
    let inv_rms = (stats[row] + *eps).sqrt().recip();
    output[idx] = input[idx] * inv_rms * *scale * gamma[gamma_idx];
}

#[cube(launch_unchecked)]
fn multihead_rms_norm_module_affine_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    gamma: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    scale: &f32,
    eps: &f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let out_idx = ABSOLUTE_POS;
    let channel = out_idx % *head_dim;
    let token = (out_idx / *head_dim) % *tokens;
    let head = (out_idx / (*head_dim * *tokens)) % *heads;
    let batch = out_idx / (*head_dim * *tokens * *heads);
    let row = (batch * *tokens + token) * *heads + head;
    if row >= *rows {
        terminate!();
    }
    let input_idx = row * *head_dim + channel;
    let gamma_idx = head * *head_dim + channel;
    let inv_rms = (stats[row] + *eps).sqrt().recip();
    output[out_idx] = input[input_idx] * inv_rms * *scale * gamma[gamma_idx];
}

#[cube(launch_unchecked)]
fn multihead_rms_norm_rope_coords_pair_affine_kernel(
    input: &Array<f32>,
    stats: &Array<f32>,
    gamma: &Array<f32>,
    coords: &Array<i32>,
    pair_freq: &Array<f32>,
    pair_axis: &Array<i32>,
    output: &mut Array<f32>,
    rows: &usize,
    tokens: &usize,
    heads: &usize,
    head_dim: &usize,
    pairs: &usize,
    scale: &f32,
    eps: &f32,
) {
    let total_pairs = *rows * *pairs;
    if ABSOLUTE_POS >= total_pairs {
        terminate!();
    }
    let pair_idx = ABSOLUTE_POS;
    let row = pair_idx / *pairs;
    if row >= *rows {
        terminate!();
    }
    let pair = pair_idx % *pairs;
    let pair_channel = pair * 2;
    let head = row % *heads;
    let token = (row / *heads) % *tokens;
    let base = row * *head_dim;
    let inv_rms = (stats[row] + *eps).sqrt().recip();

    let even_idx = base + pair_channel;
    let odd_idx = even_idx + 1;
    let gamma_base = head * *head_dim;
    let even = input[even_idx] * inv_rms * *scale * gamma[gamma_base + pair_channel];
    let odd = input[odd_idx] * inv_rms * *scale * gamma[gamma_base + pair_channel + 1];

    let axis = pair_axis[pair];
    let mut phase = 0.0f32;
    if axis >= 0 {
        let coord_base = token * 3;
        let coord = if axis == 0 {
            coords[coord_base]
        } else if axis == 1 {
            coords[coord_base + 1]
        } else {
            coords[coord_base + 2]
        };
        phase = f32::cast_from(coord) * pair_freq[pair];
    }
    let c = phase.cos();
    let s = phase.sin();
    output[even_idx] = even * c - odd * s;
    output[odd_idx] = even * s + odd * c;
}

/// Rotate RoPE pairs in one device pass.
///
/// This replaces a long chain of reshape/slice/cat tensor ops in sparse-flow
/// attention hot paths with one dispatch on WGPU.
pub fn rope_rotate_pairs_wgpu(
    x: BurnTensor<DefaultWgpuBackend, 4>,
    cos: BurnTensor<DefaultWgpuBackend, 4>,
    sin: BurnTensor<DefaultWgpuBackend, 4>,
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = x.dtype().into();
    let x = cast_float_tensor_if_needed(x, burn::tensor::FloatDType::F32);
    let cos = cast_float_tensor_if_needed(cos, burn::tensor::FloatDType::F32);
    let sin = cast_float_tensor_if_needed(sin, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = x.dims();
    if head_dim % 2 != 0 {
        return Err(format!("rope rotate expects even head_dim, got {head_dim}"));
    }
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(x);
    }
    let pairs = head_dim / 2;
    let [cos_batch, cos_tokens, cos_heads, cos_pairs] = cos.dims();
    let [sin_batch, sin_tokens, sin_heads, sin_pairs] = sin.dims();
    if cos_batch != 1 || cos_tokens != tokens || cos_heads != 1 || cos_pairs != pairs {
        return Err(format!(
            "rope rotate cos tensor dims mismatch: got=[{cos_batch},{cos_tokens},{cos_heads},{cos_pairs}] expected=[1,{tokens},1,{pairs}]"
        ));
    }
    if sin_batch != 1 || sin_tokens != tokens || sin_heads != 1 || sin_pairs != pairs {
        return Err(format!(
            "rope rotate sin tensor dims mismatch: got=[{sin_batch},{sin_tokens},{sin_heads},{sin_pairs}] expected=[1,{tokens},1,{pairs}]"
        ));
    }

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "rope rotate row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "rope rotate output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "rope rotate output byte size overflow".to_string())?;

    let x_p = x.reshape([rows, head_dim]).into_primitive().tensor();
    let cos_p = cos.reshape([tokens, pairs]).into_primitive().tensor();
    let sin_p = sin.reshape([tokens, pairs]).into_primitive().tensor();
    let output = CubeTensor::new_contiguous(
        x_p.client.clone(),
        x_p.device.clone(),
        Shape::new([rows, head_dim]),
        x_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&x_p.client, output_elements, cube_dim);
    unsafe {
        rope_rotate_pairs_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &x_p.client,
            cube_count,
            cube_dim,
            x_p.clone().into_array_arg(),
            cos_p.clone().into_array_arg(),
            sin_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            pairs,
        )
        .map_err(|err| format!("rope_rotate_pairs_kernel launch failed: {err:?}"))?;
    }
    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(TensorPrimitive::Float(output))
            .reshape([batch, tokens, heads, head_dim]),
        input_dtype,
    ))
}

/// Rotate RoPE pairs from phase tensor `[tokens, pairs]` in one device pass.
///
/// This avoids separate `cos` and `sin` tensor materialization on sparse-flow
/// token-coordinate paths.
pub fn rope_rotate_pairs_from_phase_wgpu(
    x: BurnTensor<DefaultWgpuBackend, 4>,
    phase: BurnTensor<DefaultWgpuBackend, 2>,
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = x.dtype().into();
    let x = cast_float_tensor_if_needed(x, burn::tensor::FloatDType::F32);
    let phase = cast_float_tensor_if_needed(phase, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = x.dims();
    if head_dim % 2 != 0 {
        return Err(format!("rope rotate expects even head_dim, got {head_dim}"));
    }
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(x);
    }
    let pairs = head_dim / 2;
    let [phase_tokens, phase_pairs] = phase.dims();
    if phase_tokens != tokens || phase_pairs != pairs {
        return Err(format!(
            "rope rotate phase tensor dims mismatch: got=[{phase_tokens},{phase_pairs}] expected=[{tokens},{pairs}]"
        ));
    }

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "rope rotate row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "rope rotate output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "rope rotate output byte size overflow".to_string())?;

    let x_p = x.reshape([rows, head_dim]).into_primitive().tensor();
    let phase_p = phase.reshape([tokens * pairs]).into_primitive().tensor();
    let output = CubeTensor::new_contiguous(
        x_p.client.clone(),
        x_p.device.clone(),
        Shape::new([rows, head_dim]),
        x_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&x_p.client, output_elements, cube_dim);
    unsafe {
        rope_rotate_pairs_phase_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &x_p.client,
            cube_count,
            cube_dim,
            x_p.clone().into_array_arg(),
            phase_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            pairs,
        )
        .map_err(|err| format!("rope_rotate_pairs_phase_kernel launch failed: {err:?}"))?;
    }
    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(TensorPrimitive::Float(output))
            .reshape([batch, tokens, heads, head_dim]),
        input_dtype,
    ))
}

fn rope_pair_layout_params(pairs: usize, rope_freq: [f32; 2]) -> (Vec<f32>, Vec<i32>) {
    let freq_dim = (pairs / 3).max(1);
    let mut pair_freq = vec![0.0f32; pairs];
    let mut pair_axis = vec![-1i32; pairs];
    for pair in 0..pairs {
        let (axis, freq_idx) = if pair < freq_dim {
            (0i32, pair)
        } else if pair < freq_dim * 2 {
            (1i32, pair - freq_dim)
        } else if pair < freq_dim * 3 {
            (2i32, pair - freq_dim * 2)
        } else {
            (-1i32, 0usize)
        };
        pair_axis[pair] = axis;
        if axis >= 0 {
            let exp = freq_idx as f32 / freq_dim as f32;
            pair_freq[pair] = rope_freq[0] / rope_freq[1].powf(exp);
        }
    }
    (pair_freq, pair_axis)
}

/// Rotate RoPE pairs directly from token coords `[tokens,3]` in one device pass.
///
/// This removes intermediate phase/cos/sin tensor materialization from the
/// sparse-flow token-coordinate RoPE path.
pub fn rope_rotate_pairs_from_coords_wgpu(
    x: BurnTensor<DefaultWgpuBackend, 4>,
    coords: BurnTensor<DefaultWgpuBackend, 2, Int>,
    rope_freq: [f32; 2],
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = x.dtype().into();
    let x = cast_float_tensor_if_needed(x, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = x.dims();
    if head_dim % 2 != 0 {
        return Err(format!("rope rotate expects even head_dim, got {head_dim}"));
    }
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(x);
    }
    let [coord_rows, coord_cols] = coords.dims();
    if coord_rows != tokens || coord_cols != 3 {
        return Err(format!(
            "rope rotate coords tensor dims mismatch: got=[{coord_rows},{coord_cols}] expected=[{tokens},3]"
        ));
    }
    let pairs = head_dim / 2;
    let (pair_freq, pair_axis) = rope_pair_layout_params(pairs, rope_freq);

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "rope rotate row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "rope rotate output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "rope rotate output byte size overflow".to_string())?;

    let device = x.device();
    let pair_freq_t =
        BurnTensor::<DefaultWgpuBackend, 1>::from_floats(pair_freq.as_slice(), &device);
    let pair_axis_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(pair_axis, [pairs]),
        &device,
    );

    let x_p = x.reshape([rows, head_dim]).into_primitive().tensor();
    let coords_p = coords.reshape([tokens * 3]).into_primitive();
    let pair_freq_p = pair_freq_t.into_primitive().tensor();
    let pair_axis_p = pair_axis_t.into_primitive();
    let output = CubeTensor::new_contiguous(
        x_p.client.clone(),
        x_p.device.clone(),
        Shape::new([rows, head_dim]),
        x_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&x_p.client, output_elements, cube_dim);
    unsafe {
        rope_rotate_pairs_coords_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &x_p.client,
            cube_count,
            cube_dim,
            x_p.clone().into_array_arg(),
            coords_p.clone().into_array_arg(),
            pair_freq_p.clone().into_array_arg(),
            pair_axis_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
        )
        .map_err(|err| format!("rope_rotate_pairs_coords_kernel launch failed: {err:?}"))?;
    }
    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(TensorPrimitive::Float(output))
            .reshape([batch, tokens, heads, head_dim]),
        input_dtype,
    ))
}

/// Compute `output = input * weight^T + bias` for skinny output heads.
///
/// Intended for decode hotspots where `out_channels` is small (for example <= 8)
/// and row count is large. A dedicated kernel avoids the high dispatch overhead
/// seen with multi-pass tensor-op reduction formulations in this regime.
pub fn linear_skinny_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 2>,
    weight: BurnTensor<DefaultWgpuBackend, 2>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let [rows, in_channels] = input.dims();
    let [out_channels, weight_in_channels] = weight.dims();
    let [bias_channels] = bias.dims();
    if in_channels != weight_in_channels {
        return Err(format!(
            "skinny linear input/weight mismatch: input_in_channels={in_channels} weight_in_channels={weight_in_channels}"
        ));
    }
    if out_channels != bias_channels {
        return Err(format!(
            "skinny linear bias mismatch: out_channels={out_channels} bias_len={bias_channels}"
        ));
    }
    if rows == 0 || out_channels == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [rows, out_channels],
            &input.device(),
        ));
    }

    let output_elements = rows
        .checked_mul(out_channels)
        .ok_or_else(|| "skinny linear output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "skinny linear output byte size overflow".to_string())?;

    let input_p = input.into_primitive().tensor();
    let weight_p = weight
        .reshape([out_channels * in_channels])
        .into_primitive()
        .tensor();
    let bias_p = bias.into_primitive().tensor();
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, out_channels]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        linear_skinny_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            weight_p.clone().into_array_arg(),
            bias_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            in_channels,
            out_channels,
        )
        .map_err(|err| format!("linear_skinny_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(
        TensorPrimitive::Float(output),
    ))
}

/// Fused row-wise layer-norm with affine parameters on 2D tensors.
///
/// Computes per-row mean/variance then applies:
/// `y = (x - mean) / sqrt(var + eps) * weight + bias`.
pub fn layer_norm_affine_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 2>,
    weight: BurnTensor<DefaultWgpuBackend, 1>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let weight = cast_float_tensor_if_needed(weight, burn::tensor::FloatDType::F32);
    let bias = cast_float_tensor_if_needed(bias, burn::tensor::FloatDType::F32);
    let [rows, channels] = input.dims();
    if rows == 0 || channels == 0 {
        return Ok(input);
    }
    let [weight_channels] = weight.dims();
    let [bias_channels] = bias.dims();
    if channels != weight_channels {
        return Err(format!(
            "layer norm weight mismatch: channels={channels} weight_len={weight_channels}"
        ));
    }
    if channels != bias_channels {
        return Err(format!(
            "layer norm bias mismatch: channels={channels} bias_len={bias_channels}"
        ));
    }

    let stats_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "layer norm stats size overflow".to_string())?;
    let stats_bytes = stats_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(channels)
        .ok_or_else(|| "layer norm output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm output byte size overflow".to_string())?;

    let input_p = input.into_primitive().tensor();
    let weight_p = weight.into_primitive().tensor();
    let bias_p = bias.into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, 2]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, channels]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    launch_layer_norm_row_stats(&input_p, &stats, rows, channels, cube_dim)?;

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        layer_norm_affine_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            weight_p.clone().into_array_arg(),
            bias_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            channels,
            eps,
        )
        .map_err(|err| format!("layer_norm_affine_kernel launch failed: {err:?}"))?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

/// Fused row-wise layer-norm + affine + SiLU on 2D tensors.
pub fn layer_norm_affine_silu_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 2>,
    weight: BurnTensor<DefaultWgpuBackend, 1>,
    bias: BurnTensor<DefaultWgpuBackend, 1>,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let weight = cast_float_tensor_if_needed(weight, burn::tensor::FloatDType::F32);
    let bias = cast_float_tensor_if_needed(bias, burn::tensor::FloatDType::F32);
    let [rows, channels] = input.dims();
    if rows == 0 || channels == 0 {
        return Ok(input);
    }
    let [weight_channels] = weight.dims();
    let [bias_channels] = bias.dims();
    if channels != weight_channels {
        return Err(format!(
            "layer norm silu weight mismatch: channels={channels} weight_len={weight_channels}"
        ));
    }
    if channels != bias_channels {
        return Err(format!(
            "layer norm silu bias mismatch: channels={channels} bias_len={bias_channels}"
        ));
    }

    let stats_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "layer norm silu stats size overflow".to_string())?;
    let stats_bytes = stats_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm silu stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(channels)
        .ok_or_else(|| "layer norm silu output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm silu output byte size overflow".to_string())?;

    let input_p = input.into_primitive().tensor();
    let weight_p = weight.into_primitive().tensor();
    let bias_p = bias.into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, 2]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, channels]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    launch_layer_norm_row_stats(&input_p, &stats, rows, channels, cube_dim)?;

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        layer_norm_affine_silu_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            weight_p.clone().into_array_arg(),
            bias_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            channels,
            eps,
        )
        .map_err(|err| format!("layer_norm_affine_silu_kernel launch failed: {err:?}"))?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 2>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

/// Fused row-wise layer-norm plus adaptive modulation on 3D tensors.
///
/// Computes per-token layer norm then applies:
/// `y = norm(x) * (1 + scale[batch, channel]) + shift[batch, channel]`.
pub fn layer_norm_modulated_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 3>,
    scale: BurnTensor<DefaultWgpuBackend, 3>,
    shift: BurnTensor<DefaultWgpuBackend, 3>,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 3>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let [batch, tokens, channels] = input.dims();
    if batch == 0 || tokens == 0 || channels == 0 {
        return Ok(input);
    }
    let [scale_batch, scale_tokens, scale_channels] = scale.dims();
    let [shift_batch, shift_tokens, shift_channels] = shift.dims();
    if scale_batch != batch || scale_tokens != 1 || scale_channels != channels {
        return Err(format!(
            "layer norm modulation scale mismatch: scale=[{scale_batch},{scale_tokens},{scale_channels}] expected=[{batch},1,{channels}]"
        ));
    }
    if shift_batch != batch || shift_tokens != 1 || shift_channels != channels {
        return Err(format!(
            "layer norm modulation shift mismatch: shift=[{shift_batch},{shift_tokens},{shift_channels}] expected=[{batch},1,{channels}]"
        ));
    }

    let scale_dtype: burn::tensor::FloatDType = scale.dtype().into();
    let shift_dtype: burn::tensor::FloatDType = shift.dtype().into();
    if input_dtype == burn::tensor::FloatDType::F16
        && scale_dtype == burn::tensor::FloatDType::F16
        && shift_dtype == burn::tensor::FloatDType::F16
    {
        return layer_norm_modulated_forward_wgpu_f16(input, scale, shift, eps);
    }

    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let scale = cast_float_tensor_if_needed(scale, burn::tensor::FloatDType::F32);
    let shift = cast_float_tensor_if_needed(shift, burn::tensor::FloatDType::F32);

    let rows = batch
        .checked_mul(tokens)
        .ok_or_else(|| "layer norm modulation row count overflow".to_string())?;
    let stats_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "layer norm modulation stats size overflow".to_string())?;
    let stats_bytes = stats_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm modulation stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(channels)
        .ok_or_else(|| "layer norm modulation output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm modulation output byte size overflow".to_string())?;

    let input_p = input.reshape([rows, channels]).into_primitive().tensor();
    let scale_p = scale.reshape([batch * channels]).into_primitive().tensor();
    let shift_p = shift.reshape([batch * channels]).into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, 2]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, channels]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    launch_layer_norm_row_stats(&input_p, &stats, rows, channels, cube_dim)?;

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        layer_norm_modulated_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            scale_p.clone().into_array_arg(),
            shift_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            channels,
            eps,
        )
        .map_err(|err| format!("layer_norm_modulated_kernel launch failed: {err:?}"))?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 3>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

fn layer_norm_modulated_forward_wgpu_f16(
    input: BurnTensor<DefaultWgpuBackend, 3>,
    scale: BurnTensor<DefaultWgpuBackend, 3>,
    shift: BurnTensor<DefaultWgpuBackend, 3>,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 3>, String> {
    let [batch, tokens, channels] = input.dims();
    let rows = batch
        .checked_mul(tokens)
        .ok_or_else(|| "layer norm modulation f16 row count overflow".to_string())?;
    let stats_elements = rows
        .checked_mul(2)
        .ok_or_else(|| "layer norm modulation f16 stats size overflow".to_string())?;
    let stats_bytes = stats_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "layer norm modulation f16 stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(channels)
        .ok_or_else(|| "layer norm modulation f16 output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f16>())
        .ok_or_else(|| "layer norm modulation f16 output byte size overflow".to_string())?;

    let input_p = input.reshape([rows, channels]).into_primitive().tensor();
    let scale_p = scale.reshape([batch * channels]).into_primitive().tensor();
    let shift_p = shift.reshape([batch * channels]).into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows, 2]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, channels]),
        input_p.client.empty(output_bytes),
        DType::F16,
    );

    let cube_dim = resolve_cube_dim();
    launch_layer_norm_row_stats_f16(&input_p, &stats, rows, channels, cube_dim)?;

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        layer_norm_modulated_f16_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            scale_p.clone().into_array_arg(),
            shift_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            channels,
            eps,
        )
        .map_err(|err| format!("layer_norm_modulated_f16_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::<DefaultWgpuBackend, 3>::from_primitive(
        TensorPrimitive::Float(output),
    ))
}

/// Fused multi-head RMS norm with affine gamma on `[batch,tokens,heads,head_dim]`.
///
/// Matches the TRELLIS sparse-flow Q/K norm convention:
/// `y = x / sqrt(sum(x^2) + eps) * sqrt(head_dim) * gamma[head, dim]`.
pub fn multihead_rms_norm_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 4>,
    gamma: BurnTensor<DefaultWgpuBackend, 2>,
    scale: f32,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let gamma = cast_float_tensor_if_needed(gamma, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = input.dims();
    if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(input);
    }
    let [gamma_heads, gamma_head_dim] = gamma.dims();
    if gamma_heads != heads || gamma_head_dim != head_dim {
        return Err(format!(
            "multihead rms norm gamma mismatch: gamma=[{gamma_heads},{gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "multihead rms norm row count overflow".to_string())?;
    let stats_bytes = rows
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead rms norm stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "multihead rms norm output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead rms norm output byte size overflow".to_string())?;

    let input_p = input.reshape([rows, head_dim]).into_primitive().tensor();
    let gamma_p = gamma.reshape([heads * head_dim]).into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, heads, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let row_cube_count = calculate_cube_count_elemwise(&input_p.client, rows, cube_dim);
    unsafe {
        multihead_rms_norm_row_stats_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            row_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            rows,
            head_dim,
        )
        .map_err(|err| format!("multihead_rms_norm_row_stats_kernel launch failed: {err:?}"))?;
    }

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        multihead_rms_norm_affine_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            gamma_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            heads,
            head_dim,
            scale,
            eps,
        )
        .map_err(|err| format!("multihead_rms_norm_affine_kernel launch failed: {err:?}"))?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

/// Fused multi-head RMS norm with output in module-attention layout.
///
/// Input is `[batch,tokens,heads,head_dim]`; output is
/// `[batch,heads,tokens,head_dim]`. This preserves the same math as
/// [`multihead_rms_norm_forward_wgpu`] followed by `permute([0,2,1,3])`,
/// while avoiding a separate layout materialization before attention.
pub fn multihead_rms_norm_module_forward_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 4>,
    gamma: BurnTensor<DefaultWgpuBackend, 2>,
    scale: f32,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let gamma = cast_float_tensor_if_needed(gamma, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = input.dims();
    if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(input.reshape([batch, heads, tokens, head_dim]));
    }
    let [gamma_heads, gamma_head_dim] = gamma.dims();
    if gamma_heads != heads || gamma_head_dim != head_dim {
        return Err(format!(
            "multihead rms norm module gamma mismatch: gamma=[{gamma_heads},{gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "multihead rms norm module row count overflow".to_string())?;
    let stats_bytes = rows
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead rms norm module stats byte size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "multihead rms norm module output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead rms norm module output byte size overflow".to_string())?;

    let input_p = input.reshape([rows, head_dim]).into_primitive().tensor();
    let gamma_p = gamma.reshape([heads * head_dim]).into_primitive().tensor();
    let stats = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([rows]),
        input_p.client.empty(stats_bytes),
        DType::F32,
    );
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, heads, tokens, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let row_cube_count = calculate_cube_count_elemwise(&input_p.client, rows, cube_dim);
    unsafe {
        multihead_rms_norm_row_stats_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            row_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            rows,
            head_dim,
        )
        .map_err(|err| format!("multihead_rms_norm_row_stats_kernel launch failed: {err:?}"))?;
    }

    let output_cube_count =
        calculate_cube_count_elemwise(&input_p.client, output_elements, cube_dim);
    unsafe {
        multihead_rms_norm_module_affine_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input_p.client,
            output_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            stats.clone().into_array_arg(),
            gamma_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            scale,
            eps,
        )
        .map_err(|err| format!("multihead_rms_norm_module_affine_kernel launch failed: {err:?}"))?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

/// Fused multi-head RMS norm plus coordinate RoPE rotation.
///
/// Matches applying [`multihead_rms_norm_forward_wgpu`] first and
/// [`rope_rotate_pairs_from_coords_wgpu`] second, but avoids materializing the
/// normalized intermediate tensor.
pub fn multihead_rms_norm_rope_from_coords_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 4>,
    gamma: BurnTensor<DefaultWgpuBackend, 2>,
    coords: BurnTensor<DefaultWgpuBackend, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Result<BurnTensor<DefaultWgpuBackend, 4>, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let gamma = cast_float_tensor_if_needed(gamma, burn::tensor::FloatDType::F32);
    let [batch, tokens, heads, head_dim] = input.dims();
    if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
        return Ok(input);
    }
    if head_dim % 2 != 0 {
        return Err(format!(
            "multihead rms norm rope expects even head_dim, got {head_dim}"
        ));
    }
    let [gamma_heads, gamma_head_dim] = gamma.dims();
    if gamma_heads != heads || gamma_head_dim != head_dim {
        return Err(format!(
            "multihead rms norm rope gamma mismatch: gamma=[{gamma_heads},{gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let [coord_rows, coord_cols] = coords.dims();
    if coord_rows != tokens || coord_cols != 3 {
        return Err(format!(
            "multihead rms norm rope coords mismatch: coords=[{coord_rows},{coord_cols}] expected=[{tokens},3]"
        ));
    }

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "multihead rms norm rope row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "multihead rms norm rope output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead rms norm rope output byte size overflow".to_string())?;
    let pairs = head_dim / 2;
    let input_p = input.reshape([rows, head_dim]).into_primitive().tensor();
    let gamma_p = gamma.reshape([heads * head_dim]).into_primitive().tensor();
    let coords_p = coords.reshape([tokens * 3]).into_primitive();
    let output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, heads, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let row_cube_count = calculate_cube_count_elemwise(&input_p.client, rows, cube_dim);
    unsafe {
        multihead_rms_norm_rope_coords_row_affine_kernel::launch_unchecked::<
            burn_wgpu::WgpuRuntime,
        >(
            &input_p.client,
            row_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            gamma_p.clone().into_array_arg(),
            coords_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            pairs,
            rope_freq[0],
            rope_freq[1].ln(),
            scale,
            eps,
        )
        .map_err(|err| {
            format!("multihead_rms_norm_rope_coords_row_affine_kernel launch failed: {err:?}")
        })?;
    }

    Ok(cast_float_tensor_if_needed(
        BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(output)),
        input_dtype,
    ))
}

/// Fused Q/K RMS norm + coordinate RoPE directly from packed QKV.
///
/// The input layout is `[batch, tokens, 3, heads, head_dim]`, matching the
/// TRELLIS sparse-flow self-attention projection before Q/K/V slicing. This
/// avoids launching separate Q and K RMS+RoPE kernels in the dominant flow path.
pub fn multihead_qk_rms_norm_rope_from_qkv_coords_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 5>,
    q_gamma: BurnTensor<DefaultWgpuBackend, 2>,
    k_gamma: BurnTensor<DefaultWgpuBackend, 2>,
    coords: BurnTensor<DefaultWgpuBackend, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Result<
    (
        BurnTensor<DefaultWgpuBackend, 4>,
        BurnTensor<DefaultWgpuBackend, 4>,
    ),
    String,
> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let q_gamma = cast_float_tensor_if_needed(q_gamma, burn::tensor::FloatDType::F32);
    let k_gamma = cast_float_tensor_if_needed(k_gamma, burn::tensor::FloatDType::F32);
    let [batch, tokens, qkv, heads, head_dim] = input.dims();
    if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
        let q = BurnTensor::<DefaultWgpuBackend, 4>::zeros(
            [batch, tokens, heads, head_dim],
            &input.device(),
        );
        let k = BurnTensor::<DefaultWgpuBackend, 4>::zeros(
            [batch, tokens, heads, head_dim],
            &input.device(),
        );
        return Ok((q, k));
    }
    if qkv != 3 {
        return Err(format!(
            "multihead qk rms norm rope expects qkv dimension 3, got {qkv}"
        ));
    }
    if head_dim % 2 != 0 {
        return Err(format!(
            "multihead qk rms norm rope expects even head_dim, got {head_dim}"
        ));
    }
    let [q_gamma_heads, q_gamma_head_dim] = q_gamma.dims();
    if q_gamma_heads != heads || q_gamma_head_dim != head_dim {
        return Err(format!(
            "multihead q rms norm rope gamma mismatch: gamma=[{q_gamma_heads},{q_gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let [k_gamma_heads, k_gamma_head_dim] = k_gamma.dims();
    if k_gamma_heads != heads || k_gamma_head_dim != head_dim {
        return Err(format!(
            "multihead k rms norm rope gamma mismatch: gamma=[{k_gamma_heads},{k_gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let [coord_rows, coord_cols] = coords.dims();
    if coord_rows != tokens || coord_cols != 3 {
        return Err(format!(
            "multihead qk rms norm rope coords mismatch: coords=[{coord_rows},{coord_cols}] expected=[{tokens},3]"
        ));
    }

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "multihead qk rms norm rope row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "multihead qk rms norm rope output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "multihead qk rms norm rope output byte size overflow".to_string())?;
    let pairs = head_dim / 2;
    let input_p = input
        .reshape([batch, tokens, qkv, heads, head_dim])
        .into_primitive()
        .tensor();
    let q_gamma_p = q_gamma
        .reshape([heads * head_dim])
        .into_primitive()
        .tensor();
    let k_gamma_p = k_gamma
        .reshape([heads * head_dim])
        .into_primitive()
        .tensor();
    let coords_p = coords.reshape([tokens * 3]).into_primitive();
    let q_output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, heads, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );
    let k_output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, tokens, heads, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let row_cube_count = calculate_cube_count_elemwise(&input_p.client, rows, cube_dim);
    unsafe {
        multihead_qk_rms_norm_rope_coords_from_qkv_kernel::launch_unchecked::<
            burn_wgpu::WgpuRuntime,
        >(
            &input_p.client,
            row_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            q_gamma_p.clone().into_array_arg(),
            k_gamma_p.clone().into_array_arg(),
            coords_p.clone().into_array_arg(),
            q_output.clone().into_array_arg(),
            k_output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            pairs,
            rope_freq[0],
            rope_freq[1].ln(),
            scale,
            eps,
        )
        .map_err(|err| {
            format!("multihead_qk_rms_norm_rope_coords_from_qkv_kernel launch failed: {err:?}")
        })?;
    }

    Ok((
        cast_float_tensor_if_needed(
            BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(q_output)),
            input_dtype,
        ),
        cast_float_tensor_if_needed(
            BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(k_output)),
            input_dtype,
        ),
    ))
}

/// Fused Q/K RMS norm + coordinate RoPE and V extraction directly from packed QKV.
///
/// The returned tensors use module-attention layout `[batch, heads, tokens, head_dim]`.
/// This avoids token-major Q/K outputs followed by separate V slicing, permutation,
/// and cast/materialization in long sparse-flow self-attention blocks.
pub fn multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu(
    input: BurnTensor<DefaultWgpuBackend, 5>,
    q_gamma: BurnTensor<DefaultWgpuBackend, 2>,
    k_gamma: BurnTensor<DefaultWgpuBackend, 2>,
    coords: BurnTensor<DefaultWgpuBackend, 2, Int>,
    rope_freq: [f32; 2],
    scale: f32,
    eps: f32,
) -> Result<ModuleQkvRmsNormRopeOutput, String> {
    let input_dtype: burn::tensor::FloatDType = input.dtype().into();
    let input = cast_float_tensor_if_needed(input, burn::tensor::FloatDType::F32);
    let q_gamma = cast_float_tensor_if_needed(q_gamma, burn::tensor::FloatDType::F32);
    let k_gamma = cast_float_tensor_if_needed(k_gamma, burn::tensor::FloatDType::F32);
    let [batch, tokens, qkv, heads, head_dim] = input.dims();
    if batch == 0 || tokens == 0 || heads == 0 || head_dim == 0 {
        let q = BurnTensor::<DefaultWgpuBackend, 4>::zeros(
            [batch, heads, tokens, head_dim],
            &input.device(),
        );
        let k = BurnTensor::<DefaultWgpuBackend, 4>::zeros(
            [batch, heads, tokens, head_dim],
            &input.device(),
        );
        let v = BurnTensor::<DefaultWgpuBackend, 4>::zeros(
            [batch, heads, tokens, head_dim],
            &input.device(),
        );
        return Ok((q, k, v));
    }
    if qkv != 3 {
        return Err(format!(
            "multihead qkv module rms norm rope expects qkv dimension 3, got {qkv}"
        ));
    }
    if head_dim % 2 != 0 {
        return Err(format!(
            "multihead qkv module rms norm rope expects even head_dim, got {head_dim}"
        ));
    }
    let [q_gamma_heads, q_gamma_head_dim] = q_gamma.dims();
    if q_gamma_heads != heads || q_gamma_head_dim != head_dim {
        return Err(format!(
            "multihead q rms norm rope gamma mismatch: gamma=[{q_gamma_heads},{q_gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let [k_gamma_heads, k_gamma_head_dim] = k_gamma.dims();
    if k_gamma_heads != heads || k_gamma_head_dim != head_dim {
        return Err(format!(
            "multihead k rms norm rope gamma mismatch: gamma=[{k_gamma_heads},{k_gamma_head_dim}] expected=[{heads},{head_dim}]"
        ));
    }
    let [coord_rows, coord_cols] = coords.dims();
    if coord_rows != tokens || coord_cols != 3 {
        return Err(format!(
            "multihead qkv module rms norm rope coords mismatch: coords=[{coord_rows},{coord_cols}] expected=[{tokens},3]"
        ));
    }

    let rows = batch
        .checked_mul(tokens)
        .and_then(|value| value.checked_mul(heads))
        .ok_or_else(|| "multihead qkv module rms norm rope row count overflow".to_string())?;
    let output_elements = rows
        .checked_mul(head_dim)
        .ok_or_else(|| "multihead qkv module rms norm rope output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| {
            "multihead qkv module rms norm rope output byte size overflow".to_string()
        })?;
    let pairs = head_dim / 2;
    let input_p = input
        .reshape([batch, tokens, qkv, heads, head_dim])
        .into_primitive()
        .tensor();
    let q_gamma_p = q_gamma
        .reshape([heads * head_dim])
        .into_primitive()
        .tensor();
    let k_gamma_p = k_gamma
        .reshape([heads * head_dim])
        .into_primitive()
        .tensor();
    let coords_p = coords.reshape([tokens * 3]).into_primitive();
    let q_output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, heads, tokens, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );
    let k_output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, heads, tokens, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );
    let v_output = CubeTensor::new_contiguous(
        input_p.client.clone(),
        input_p.device.clone(),
        Shape::new([batch, heads, tokens, head_dim]),
        input_p.client.empty(output_bytes),
        DType::F32,
    );

    let cube_dim = resolve_cube_dim();
    let row_cube_count = calculate_cube_count_elemwise(&input_p.client, rows, cube_dim);
    unsafe {
        multihead_qkv_module_rms_norm_rope_coords_from_qkv_kernel::launch_unchecked::<
            burn_wgpu::WgpuRuntime,
        >(
            &input_p.client,
            row_cube_count,
            cube_dim,
            input_p.clone().into_array_arg(),
            q_gamma_p.clone().into_array_arg(),
            k_gamma_p.clone().into_array_arg(),
            coords_p.clone().into_array_arg(),
            q_output.clone().into_array_arg(),
            k_output.clone().into_array_arg(),
            v_output.clone().into_array_arg(),
            rows,
            tokens,
            heads,
            head_dim,
            pairs,
            rope_freq[0],
            rope_freq[1].ln(),
            scale,
            eps,
        )
        .map_err(|err| {
            format!(
                "multihead_qkv_module_rms_norm_rope_coords_from_qkv_kernel launch failed: {err:?}"
            )
        })?;
    }

    Ok((
        cast_float_tensor_if_needed(
            BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(q_output)),
            input_dtype,
        ),
        cast_float_tensor_if_needed(
            BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(k_output)),
            input_dtype,
        ),
        cast_float_tensor_if_needed(
            BurnTensor::<DefaultWgpuBackend, 4>::from_primitive(TensorPrimitive::Float(v_output)),
            input_dtype,
        ),
    ))
}

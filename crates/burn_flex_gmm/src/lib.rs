use std::collections::HashMap;

pub const INVALID_NEIGHBOR_ROW: i32 = -1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseSubmConvConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_d: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub in_channels_per_group: usize,
    pub out_channels_per_group: usize,
    pub groups: usize,
    pub axis_order: [usize; 3],
    pub axis_sign: [i32; 3],
}

#[derive(Clone, Copy, Debug)]
pub struct SparseSubmConvWeights<'a> {
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparsePatchify3dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub tubelet_size: usize,
    pub patch_h: usize,
    pub patch_w: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct SparsePatchify3dWeights<'a> {
    /// Dense 3D patch projection weights in `[out_channels, in_channels, tubelet, patch_h, patch_w]`.
    pub weight: &'a [f32],
    pub bias: &'a [f32],
}

#[derive(Clone, Debug)]
struct KernelLayout {
    offsets: Vec<[i32; 3]>,
    rows: usize,
}

#[cfg(feature = "wgpu-kernel")]
pub mod wgpu_patchify;
#[cfg(feature = "wgpu-kernel")]
pub use wgpu_patchify as wgpu;

pub fn kernel_rows(config: &SparseSubmConvConfig) -> Result<usize, String> {
    validate_config(config)?;
    let rows = config
        .kernel_d
        .checked_mul(config.kernel_h)
        .and_then(|value| value.checked_mul(config.kernel_w))
        .ok_or_else(|| "sparse conv kernel rows overflow".to_string())?;
    Ok(rows)
}

pub fn build_neighbor_rows(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
) -> Result<Vec<i32>, String> {
    validate_config(config)?;
    if coords.len() > i32::MAX as usize {
        return Err("sparse conv coord row count exceeds i32::MAX".to_string());
    }
    if coords.is_empty() {
        return Ok(Vec::new());
    }

    let rows = coords.len();
    let kernel = kernel_layout(config);
    let mut coord_to_row = HashMap::with_capacity(rows.saturating_mul(2));
    for (row_idx, coord) in coords.iter().copied().enumerate() {
        coord_to_row.insert(coord, row_idx as i32);
    }

    let mut neighbor_rows = vec![INVALID_NEIGHBOR_ROW; rows * kernel.rows];
    for (out_row, coord) in coords.iter().copied().enumerate() {
        let ox = coord[1] as i32;
        let oy = coord[2] as i32;
        let oz = coord[3] as i32;
        let batch = coord[0];
        for (kernel_idx, offset) in kernel.offsets.iter().copied().enumerate() {
            let nx = ox + offset[0];
            let ny = oy + offset[1];
            let nz = oz + offset[2];
            if nx < 0 || ny < 0 || nz < 0 {
                continue;
            }
            let neighbor = [batch, nx as u32, ny as u32, nz as u32];
            if let Some(in_row) = coord_to_row.get(&neighbor).copied() {
                neighbor_rows[out_row * kernel.rows + kernel_idx] = in_row;
            }
        }
    }

    Ok(neighbor_rows)
}

pub fn sparse_subm_conv_forward_flex(
    config: &SparseSubmConvConfig,
    weights: SparseSubmConvWeights<'_>,
    coords: &[[u32; 4]],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    validate_shapes(config, weights, coords, input)?;
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }

    let neighbor_rows = build_neighbor_rows(config, coords)?;
    sparse_subm_conv_forward_flex_precomputed(
        config,
        weights,
        input,
        neighbor_rows.as_slice(),
        None,
    )
}

pub fn sparse_patchify3d_forward_flex(
    config: &SparsePatchify3dConfig,
    weights: SparsePatchify3dWeights<'_>,
    coords: &[[u32; 4]],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    let batch_count = validate_sparse_patchify3d_shapes(config, weights, input)?;
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    validate_sparse_patchify3d_coords(config, coords, batch_count)?;

    let k = sparse_patchify3d_inner_len(config)?;
    let packed_weight = pack_sparse_patchify3d_weight(config, weights.weight)?;
    let mut gathered = vec![0.0f32; rows * k];
    let mut output = vec![0.0f32; rows * config.out_channels];

    for row in 0..rows {
        let out_base = row * config.out_channels;
        output[out_base..out_base + config.out_channels].copy_from_slice(weights.bias);

        let [batch, tubelet, patch_row, patch_col] = coords[row];
        let batch = batch as usize;
        let t0 = tubelet as usize * config.tubelet_size;
        let y0 = patch_row as usize * config.patch_h;
        let x0 = patch_col as usize * config.patch_w;
        let row_base = row * k;
        let mut dst = row_base;
        for c in 0..config.in_channels {
            for dt in 0..config.tubelet_size {
                let t = t0 + dt;
                for py in 0..config.patch_h {
                    let y = y0 + py;
                    for px in 0..config.patch_w {
                        let x = x0 + px;
                        let src = (((batch * config.in_channels + c) * config.frames + t)
                            * config.height
                            + y)
                            * config.width
                            + x;
                        gathered[dst] = input[src];
                        dst += 1;
                    }
                }
            }
        }
    }

    // output[rows, out_channels] += gathered[rows, k] @ packed_weight[k, out_channels]
    unsafe {
        matrixmultiply::sgemm(
            rows,
            k,
            config.out_channels,
            1.0,
            gathered.as_ptr(),
            k as isize,
            1,
            packed_weight.as_ptr(),
            config.out_channels as isize,
            1,
            1.0,
            output.as_mut_ptr(),
            config.out_channels as isize,
            1,
        );
    }

    Ok(output)
}

pub fn pack_sparse_patchify3d_weight(
    config: &SparsePatchify3dConfig,
    weight: &[f32],
) -> Result<Vec<f32>, String> {
    validate_sparse_patchify3d_config(config)?;
    let expected_weight = expected_sparse_patchify3d_weight_len(config)?;
    if weight.len() != expected_weight {
        return Err(format!(
            "sparse patchify weight len mismatch: got {} expected {}",
            weight.len(),
            expected_weight
        ));
    }

    let k = sparse_patchify3d_inner_len(config)?;
    let mut packed = vec![0.0f32; k * config.out_channels];
    for out_channel in 0..config.out_channels {
        let mut k_idx = 0usize;
        for c in 0..config.in_channels {
            for dt in 0..config.tubelet_size {
                for py in 0..config.patch_h {
                    for px in 0..config.patch_w {
                        let src =
                            ((((out_channel * config.in_channels + c) * config.tubelet_size + dt)
                                * config.patch_h
                                + py)
                                * config.patch_w)
                                + px;
                        packed[k_idx * config.out_channels + out_channel] = weight[src];
                        k_idx += 1;
                    }
                }
            }
        }
    }
    Ok(packed)
}

pub fn sparse_subm_conv_forward_flex_precomputed(
    config: &SparseSubmConvConfig,
    weights: SparseSubmConvWeights<'_>,
    input: &[f32],
    neighbor_rows: &[i32],
    packed_weight: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    validate_config(config)?;
    if weights.bias.len() != config.out_channels {
        return Err(format!(
            "sparse conv bias len mismatch: got {} expected {}",
            weights.bias.len(),
            config.out_channels
        ));
    }
    let expected_weight = expected_weight_len(config)?;
    if weights.weight.len() != expected_weight {
        return Err(format!(
            "sparse conv weight len mismatch: got {} expected {}",
            weights.weight.len(),
            expected_weight
        ));
    }

    let kernel = kernel_layout(config);
    if !neighbor_rows.len().is_multiple_of(kernel.rows.max(1)) {
        return Err(format!(
            "sparse conv neighbor row len mismatch: got {} expected a multiple of {}",
            neighbor_rows.len(),
            kernel.rows
        ));
    }
    let rows = neighbor_rows.len().checked_div(kernel.rows).unwrap_or(0);
    let expected_input = rows
        .checked_mul(config.in_channels)
        .ok_or_else(|| "sparse conv input size overflow".to_string())?;
    if input.len() != expected_input {
        return Err(format!(
            "sparse conv input len mismatch: got {} expected {}",
            input.len(),
            expected_input
        ));
    }
    if rows == 0 {
        return Ok(Vec::new());
    }

    let k_in = kernel
        .rows
        .checked_mul(config.in_channels_per_group)
        .ok_or_else(|| "k dimension overflow in sparse_subm_conv_forward_flex".to_string())?;
    let m = rows;
    let n = config.out_channels_per_group;
    let expected_packed = config
        .groups
        .checked_mul(k_in)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| "sparse conv packed weight size overflow".to_string())?;
    let trust_neighbor_rows = packed_weight.is_some();
    let owned_packed;
    let packed = if let Some(packed_weight) = packed_weight {
        if packed_weight.len() != expected_packed {
            return Err(format!(
                "sparse conv packed weight len mismatch: got {} expected {}",
                packed_weight.len(),
                expected_packed
            ));
        }
        packed_weight
    } else {
        owned_packed = pack_flex_weight(config, weights.weight)?;
        owned_packed.as_slice()
    };

    let mut output = vec![0.0f32; rows * config.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * config.out_channels;
        output[base..base + config.out_channels].copy_from_slice(weights.bias);
    }

    let mut gathered = vec![0.0f32; m * k_in];
    for group in 0..config.groups {
        gathered.fill(0.0);
        let in_group_base = group * config.in_channels_per_group;
        for out_row in 0..m {
            for kernel_idx in 0..kernel.rows {
                let in_row = neighbor_rows[out_row * kernel.rows + kernel_idx];
                if in_row == INVALID_NEIGHBOR_ROW {
                    continue;
                }
                let in_row = if trust_neighbor_rows {
                    in_row as usize
                } else {
                    let in_row = usize::try_from(in_row).map_err(|_| {
                        format!("sparse conv neighbor row index is negative: {in_row}")
                    })?;
                    if in_row >= rows {
                        return Err(format!(
                            "sparse conv neighbor row index out of bounds: {in_row} >= {rows}"
                        ));
                    }
                    in_row
                };
                let src_base = in_row * config.in_channels + in_group_base;
                let dst_base = out_row * k_in + kernel_idx * config.in_channels_per_group;
                gathered[dst_base..dst_base + config.in_channels_per_group]
                    .copy_from_slice(&input[src_base..src_base + config.in_channels_per_group]);
            }
        }

        let out_group_base = group * config.out_channels_per_group;
        let packed_group_base = group * k_in * n;
        let packed_group = &packed[packed_group_base..packed_group_base + k_in * n];

        // output[m, n] += gathered[m, k] @ packed_group[k, n]
        unsafe {
            matrixmultiply::sgemm(
                m,
                k_in,
                n,
                1.0,
                gathered.as_ptr(),
                k_in as isize,
                1,
                packed_group.as_ptr(),
                n as isize,
                1,
                1.0,
                output.as_mut_ptr().add(out_group_base),
                config.out_channels as isize,
                1,
            );
        }
    }

    Ok(output)
}

pub fn pack_flex_weight(config: &SparseSubmConvConfig, weight: &[f32]) -> Result<Vec<f32>, String> {
    validate_config(config)?;
    let expected_weight = expected_weight_len(config)?;
    if weight.len() != expected_weight {
        return Err(format!(
            "sparse conv weight len mismatch: got {} expected {}",
            weight.len(),
            expected_weight
        ));
    }

    let kernel = kernel_layout(config);
    let k_in = kernel
        .rows
        .checked_mul(config.in_channels_per_group)
        .ok_or_else(|| "k dimension overflow in pack_flex_weight".to_string())?;
    let n = config.out_channels_per_group;
    let packed_len = config
        .groups
        .checked_mul(k_in)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| "sparse conv packed weight size overflow".to_string())?;
    let mut packed_weight = vec![0.0f32; packed_len];

    for group in 0..config.groups {
        let group_base = group * k_in * n;
        for out_local in 0..n {
            let out_idx = group * config.out_channels_per_group + out_local;
            for kd in 0..config.kernel_d {
                for kh in 0..config.kernel_h {
                    for kw in 0..config.kernel_w {
                        let kernel_idx = ((kd * config.kernel_h + kh) * config.kernel_w) + kw;
                        for in_local in 0..config.in_channels_per_group {
                            let k_col = kernel_idx * config.in_channels_per_group + in_local;
                            let src_idx = (((out_idx * config.kernel_d + kd) * config.kernel_h
                                + kh)
                                * config.kernel_w
                                + kw)
                                * config.in_channels_per_group
                                + in_local;
                            packed_weight[group_base + k_col * n + out_local] = weight[src_idx];
                        }
                    }
                }
            }
        }
    }

    Ok(packed_weight)
}

pub fn sparse_subm_conv_forward_legacy(
    config: &SparseSubmConvConfig,
    weights: SparseSubmConvWeights<'_>,
    coords: &[[u32; 4]],
    input: &[f32],
) -> Result<Vec<f32>, String> {
    validate_shapes(config, weights, coords, input)?;
    let rows = coords.len();
    if rows == 0 {
        return Ok(Vec::new());
    }

    let mut output = vec![0.0f32; rows * config.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * config.out_channels;
        output[base..base + config.out_channels].copy_from_slice(weights.bias);
    }

    let mut coord_to_row = HashMap::with_capacity(rows.saturating_mul(2));
    for (row_idx, coord) in coords.iter().copied().enumerate() {
        coord_to_row.insert(coord, row_idx);
    }

    let center_d = (config.kernel_d / 2) as i32;
    let center_h = (config.kernel_h / 2) as i32;
    let center_w = (config.kernel_w / 2) as i32;
    for (out_row_idx, out_coord) in coords.iter().copied().enumerate() {
        let batch = out_coord[0];
        let ox = out_coord[1] as i32;
        let oy = out_coord[2] as i32;
        let oz = out_coord[3] as i32;
        let out_base = out_row_idx * config.out_channels;

        for kd_idx in 0..config.kernel_d {
            for kh_idx in 0..config.kernel_h {
                for kw_idx in 0..config.kernel_w {
                    let deltas = [
                        config.axis_sign[0] * (kd_idx as i32 - center_d),
                        config.axis_sign[1] * (kh_idx as i32 - center_h),
                        config.axis_sign[2] * (kw_idx as i32 - center_w),
                    ];
                    let mut spatial = [ox, oy, oz];
                    spatial[config.axis_order[0]] += deltas[0];
                    spatial[config.axis_order[1]] += deltas[1];
                    spatial[config.axis_order[2]] += deltas[2];
                    if spatial[0] < 0 || spatial[1] < 0 || spatial[2] < 0 {
                        continue;
                    }
                    let neighbor = [
                        batch,
                        spatial[0] as u32,
                        spatial[1] as u32,
                        spatial[2] as u32,
                    ];
                    let Some(in_row_idx) = coord_to_row.get(&neighbor).copied() else {
                        continue;
                    };
                    let in_row = &input
                        [in_row_idx * config.in_channels..(in_row_idx + 1) * config.in_channels];
                    for group_idx in 0..config.groups {
                        let in_group_base = group_idx * config.in_channels_per_group;
                        let out_group_base = group_idx * config.out_channels_per_group;
                        for out_local in 0..config.out_channels_per_group {
                            let out_idx = out_group_base + out_local;
                            let weight_base =
                                (((out_idx * config.kernel_d + kd_idx) * config.kernel_h + kh_idx)
                                    * config.kernel_w
                                    + kw_idx)
                                    * config.in_channels_per_group;
                            let mut accum = 0.0f32;
                            for in_local in 0..config.in_channels_per_group {
                                accum += in_row[in_group_base + in_local]
                                    * weights.weight[weight_base + in_local];
                            }
                            output[out_base + out_idx] += accum;
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn validate_shapes(
    config: &SparseSubmConvConfig,
    weights: SparseSubmConvWeights<'_>,
    coords: &[[u32; 4]],
    input: &[f32],
) -> Result<(), String> {
    validate_config(config)?;
    if weights.bias.len() != config.out_channels {
        return Err(format!(
            "sparse conv bias len mismatch: got {} expected {}",
            weights.bias.len(),
            config.out_channels
        ));
    }
    let expected_input = coords
        .len()
        .checked_mul(config.in_channels)
        .ok_or_else(|| "sparse conv input size overflow".to_string())?;
    if input.len() != expected_input {
        return Err(format!(
            "sparse conv input len mismatch: got {} expected {}",
            input.len(),
            expected_input
        ));
    }
    let expected_weight = expected_weight_len(config)?;
    if weights.weight.len() != expected_weight {
        return Err(format!(
            "sparse conv weight len mismatch: got {} expected {}",
            weights.weight.len(),
            expected_weight
        ));
    }
    Ok(())
}

fn expected_weight_len(config: &SparseSubmConvConfig) -> Result<usize, String> {
    config
        .out_channels
        .checked_mul(config.kernel_d)
        .and_then(|v| v.checked_mul(config.kernel_h))
        .and_then(|v| v.checked_mul(config.kernel_w))
        .and_then(|v| v.checked_mul(config.in_channels_per_group))
        .ok_or_else(|| "sparse conv weight size overflow".to_string())
}

fn validate_sparse_patchify3d_shapes(
    config: &SparsePatchify3dConfig,
    weights: SparsePatchify3dWeights<'_>,
    input: &[f32],
) -> Result<usize, String> {
    validate_sparse_patchify3d_config(config)?;
    if weights.bias.len() != config.out_channels {
        return Err(format!(
            "sparse patchify bias len mismatch: got {} expected {}",
            weights.bias.len(),
            config.out_channels
        ));
    }
    let expected_weight = expected_sparse_patchify3d_weight_len(config)?;
    if weights.weight.len() != expected_weight {
        return Err(format!(
            "sparse patchify weight len mismatch: got {} expected {}",
            weights.weight.len(),
            expected_weight
        ));
    }
    let per_batch = sparse_patchify3d_input_len_per_batch(config)?;
    if !input.len().is_multiple_of(per_batch) {
        return Err(format!(
            "sparse patchify input len mismatch: got {} expected a multiple of {}",
            input.len(),
            per_batch
        ));
    }
    Ok(input.len() / per_batch)
}

pub(crate) fn validate_sparse_patchify3d_config(
    config: &SparsePatchify3dConfig,
) -> Result<(), String> {
    if config.in_channels == 0 || config.out_channels == 0 {
        return Err("sparse patchify channel dimensions must be non-zero".to_string());
    }
    if config.frames == 0 || config.height == 0 || config.width == 0 {
        return Err("sparse patchify input dimensions must be non-zero".to_string());
    }
    if config.tubelet_size == 0 || config.patch_h == 0 || config.patch_w == 0 {
        return Err("sparse patchify patch dimensions must be non-zero".to_string());
    }
    if !config.frames.is_multiple_of(config.tubelet_size) {
        return Err(format!(
            "sparse patchify frames must be divisible by tubelet_size: {} % {} != 0",
            config.frames, config.tubelet_size
        ));
    }
    if !config.height.is_multiple_of(config.patch_h) {
        return Err(format!(
            "sparse patchify height must be divisible by patch_h: {} % {} != 0",
            config.height, config.patch_h
        ));
    }
    if !config.width.is_multiple_of(config.patch_w) {
        return Err(format!(
            "sparse patchify width must be divisible by patch_w: {} % {} != 0",
            config.width, config.patch_w
        ));
    }
    Ok(())
}

pub(crate) fn sparse_patchify3d_grid(
    config: &SparsePatchify3dConfig,
) -> Result<[usize; 3], String> {
    validate_sparse_patchify3d_config(config)?;
    Ok([
        config.frames / config.tubelet_size,
        config.height / config.patch_h,
        config.width / config.patch_w,
    ])
}

fn validate_sparse_patchify3d_coords(
    config: &SparsePatchify3dConfig,
    coords: &[[u32; 4]],
    batch_count: usize,
) -> Result<(), String> {
    let [grid_t, grid_h, grid_w] = sparse_patchify3d_grid(config)?;
    for (idx, [batch, tubelet, patch_row, patch_col]) in coords.iter().copied().enumerate() {
        let batch = batch as usize;
        let tubelet = tubelet as usize;
        let patch_row = patch_row as usize;
        let patch_col = patch_col as usize;
        if batch >= batch_count {
            return Err(format!(
                "sparse patchify coord row {idx} batch out of bounds: {batch} >= {batch_count}"
            ));
        }
        if tubelet >= grid_t {
            return Err(format!(
                "sparse patchify coord row {idx} tubelet out of bounds: {tubelet} >= {grid_t}"
            ));
        }
        if patch_row >= grid_h {
            return Err(format!(
                "sparse patchify coord row {idx} patch_row out of bounds: {patch_row} >= {grid_h}"
            ));
        }
        if patch_col >= grid_w {
            return Err(format!(
                "sparse patchify coord row {idx} patch_col out of bounds: {patch_col} >= {grid_w}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn sparse_patchify3d_inner_len(
    config: &SparsePatchify3dConfig,
) -> Result<usize, String> {
    validate_sparse_patchify3d_config(config)?;
    config
        .in_channels
        .checked_mul(config.tubelet_size)
        .and_then(|v| v.checked_mul(config.patch_h))
        .and_then(|v| v.checked_mul(config.patch_w))
        .ok_or_else(|| "sparse patchify inner dimension overflow".to_string())
}

fn sparse_patchify3d_input_len_per_batch(config: &SparsePatchify3dConfig) -> Result<usize, String> {
    validate_sparse_patchify3d_config(config)?;
    config
        .in_channels
        .checked_mul(config.frames)
        .and_then(|v| v.checked_mul(config.height))
        .and_then(|v| v.checked_mul(config.width))
        .ok_or_else(|| "sparse patchify input size overflow".to_string())
}

fn expected_sparse_patchify3d_weight_len(config: &SparsePatchify3dConfig) -> Result<usize, String> {
    config
        .out_channels
        .checked_mul(sparse_patchify3d_inner_len(config)?)
        .ok_or_else(|| "sparse patchify weight size overflow".to_string())
}

fn validate_config(config: &SparseSubmConvConfig) -> Result<(), String> {
    if config.in_channels == 0 || config.out_channels == 0 {
        return Err("sparse conv channel dimensions must be non-zero".to_string());
    }
    if config.kernel_d == 0 || config.kernel_h == 0 || config.kernel_w == 0 {
        return Err("sparse conv kernel dimensions must be non-zero".to_string());
    }
    if config.groups == 0 {
        return Err("sparse conv groups must be non-zero".to_string());
    }
    if config.in_channels_per_group * config.groups != config.in_channels {
        return Err("sparse conv in_channels/group mismatch".to_string());
    }
    if config.out_channels_per_group * config.groups != config.out_channels {
        return Err("sparse conv out_channels/group mismatch".to_string());
    }
    if config.axis_order[0] > 2 || config.axis_order[1] > 2 || config.axis_order[2] > 2 {
        return Err("sparse conv axis_order must be a permutation of [0,1,2]".to_string());
    }
    let mut axis_used = [false; 3];
    for axis in config.axis_order {
        if axis_used[axis] {
            return Err("sparse conv axis_order must be a permutation of [0,1,2]".to_string());
        }
        axis_used[axis] = true;
    }
    Ok(())
}

fn kernel_layout(config: &SparseSubmConvConfig) -> KernelLayout {
    let center_d = (config.kernel_d / 2) as i32;
    let center_h = (config.kernel_h / 2) as i32;
    let center_w = (config.kernel_w / 2) as i32;
    let mut offsets = Vec::with_capacity(config.kernel_d * config.kernel_h * config.kernel_w);
    for kd_idx in 0..config.kernel_d {
        for kh_idx in 0..config.kernel_h {
            for kw_idx in 0..config.kernel_w {
                let deltas = [
                    config.axis_sign[0] * (kd_idx as i32 - center_d),
                    config.axis_sign[1] * (kh_idx as i32 - center_h),
                    config.axis_sign[2] * (kw_idx as i32 - center_w),
                ];
                let mut offset = [0i32; 3];
                offset[config.axis_order[0]] = deltas[0];
                offset[config.axis_order[1]] = deltas[1];
                offset[config.axis_order[2]] = deltas[2];
                offsets.push(offset);
            }
        }
    }
    KernelLayout {
        rows: offsets.len(),
        offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SparsePatchify3dConfig, SparsePatchify3dWeights, SparseSubmConvConfig,
        SparseSubmConvWeights, build_neighbor_rows, pack_flex_weight,
        pack_sparse_patchify3d_weight, sparse_patchify3d_forward_flex,
        sparse_subm_conv_forward_flex, sparse_subm_conv_forward_flex_precomputed,
        sparse_subm_conv_forward_legacy,
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

    fn make_sparse_line_coords(count: usize) -> Vec<[u32; 4]> {
        (0..count as u32).map(|x| [0, x, 0, 0]).collect()
    }

    fn make_sparse_grid_coords(nx: u32, ny: u32, nz: u32) -> Vec<[u32; 4]> {
        let mut coords = Vec::new();
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    // Keep a sparse subset so neighborhood lookup exercises invalid rows too.
                    if (x + y + z) % 2 == 0 {
                        coords.push([0, x, y, z]);
                    }
                }
            }
        }
        coords
    }

    #[test]
    fn flex_matches_legacy_for_small_kernel() {
        let cfg = SparseSubmConvConfig {
            in_channels: 4,
            out_channels: 6,
            kernel_d: 3,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 2,
            out_channels_per_group: 3,
            groups: 2,
            axis_order: [0, 1, 2],
            axis_sign: [1, 1, 1],
        };
        let coords = make_sparse_line_coords(16);
        let mut rng = Lcg::new(42);
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
        let weights = SparseSubmConvWeights {
            weight: &weight,
            bias: &bias,
        };
        let legacy = sparse_subm_conv_forward_legacy(&cfg, weights, &coords, &input).unwrap();
        let flex = sparse_subm_conv_forward_flex(&cfg, weights, &coords, &input).unwrap();
        assert_eq!(legacy.len(), flex.len());
        for (idx, (a, b)) in legacy.iter().zip(flex.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1.0e-5,
                "mismatch at idx={idx}: legacy={a} flex={b} diff={diff}"
            );
        }
    }

    #[test]
    fn flex_handles_empty_rows() {
        let cfg = SparseSubmConvConfig {
            in_channels: 2,
            out_channels: 2,
            kernel_d: 1,
            kernel_h: 1,
            kernel_w: 1,
            in_channels_per_group: 2,
            out_channels_per_group: 2,
            groups: 1,
            axis_order: [0, 1, 2],
            axis_sign: [1, 1, 1],
        };
        let weights = SparseSubmConvWeights {
            weight: &[1.0, 0.0, 0.0, 1.0],
            bias: &[0.25, -0.5],
        };
        let out = sparse_subm_conv_forward_flex(&cfg, weights, &[], &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn flex_precomputed_path_matches_default() {
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
        let coords = make_sparse_line_coords(32);
        let mut rng = Lcg::new(73);
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
        let weights = SparseSubmConvWeights {
            weight: &weight,
            bias: &bias,
        };

        let expected = sparse_subm_conv_forward_flex(&cfg, weights, &coords, &input).unwrap();
        let neighbors = build_neighbor_rows(&cfg, &coords).unwrap();
        let packed = pack_flex_weight(&cfg, weight.as_slice()).unwrap();
        let actual = sparse_subm_conv_forward_flex_precomputed(
            &cfg,
            weights,
            input.as_slice(),
            neighbors.as_slice(),
            Some(packed.as_slice()),
        )
        .unwrap();

        assert_eq!(expected.len(), actual.len());
        for (idx, (lhs, rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= 1.0e-5,
                "mismatch at idx={idx}: expected={lhs} actual={rhs} diff={diff}"
            );
        }
    }

    #[test]
    fn flex_matches_legacy_for_axis_permutations_and_signs() {
        let axis_orders = [[0, 1, 2], [2, 1, 0], [1, 2, 0]];
        let axis_signs = [[1, 1, 1], [-1, 1, 1], [1, -1, -1]];
        let coords = make_sparse_grid_coords(6, 5, 4);
        let mut rng = Lcg::new(991);

        let in_channels = 6usize;
        let out_channels = 8usize;
        let kernel_d = 3usize;
        let kernel_h = 3usize;
        let kernel_w = 3usize;
        let in_channels_per_group = 3usize;
        let out_channels_per_group = 4usize;
        let groups = 2usize;

        let input: Vec<f32> = (0..coords.len() * in_channels)
            .map(|_| rng.next_f32())
            .collect();
        let weight_len = out_channels * kernel_d * kernel_h * kernel_w * in_channels_per_group;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..out_channels).map(|_| rng.next_f32()).collect();
        let weights = SparseSubmConvWeights {
            weight: &weight,
            bias: &bias,
        };

        for axis_order in axis_orders {
            for axis_sign in axis_signs {
                let cfg = SparseSubmConvConfig {
                    in_channels,
                    out_channels,
                    kernel_d,
                    kernel_h,
                    kernel_w,
                    in_channels_per_group,
                    out_channels_per_group,
                    groups,
                    axis_order,
                    axis_sign,
                };
                let legacy =
                    sparse_subm_conv_forward_legacy(&cfg, weights, coords.as_slice(), &input)
                        .expect("legacy");
                let flex = sparse_subm_conv_forward_flex(&cfg, weights, coords.as_slice(), &input)
                    .expect("flex");
                assert_eq!(legacy.len(), flex.len());
                for (idx, (lhs, rhs)) in legacy.iter().zip(flex.iter()).enumerate() {
                    let diff = (lhs - rhs).abs();
                    assert!(
                        diff <= 2.0e-5,
                        "axis_order={axis_order:?} axis_sign={axis_sign:?} mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn sparse_patchify3d_matches_dense_patch_reference() {
        let cfg = SparsePatchify3dConfig {
            in_channels: 3,
            out_channels: 5,
            frames: 4,
            height: 6,
            width: 8,
            tubelet_size: 2,
            patch_h: 3,
            patch_w: 4,
        };
        let coords = vec![[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1]];
        let mut rng = Lcg::new(20260509);
        let per_batch = cfg.in_channels * cfg.frames * cfg.height * cfg.width;
        let input: Vec<f32> = (0..2 * per_batch).map(|_| rng.next_f32()).collect();
        let weight_len =
            cfg.out_channels * cfg.in_channels * cfg.tubelet_size * cfg.patch_h * cfg.patch_w;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();
        let weights = SparsePatchify3dWeights {
            weight: &weight,
            bias: &bias,
        };

        let sparse = sparse_patchify3d_forward_flex(&cfg, weights, &coords, &input).unwrap();
        let mut dense_ref = Vec::with_capacity(coords.len() * cfg.out_channels);
        for [batch, tubelet, patch_row, patch_col] in coords.iter().copied() {
            let batch = batch as usize;
            let t0 = tubelet as usize * cfg.tubelet_size;
            let y0 = patch_row as usize * cfg.patch_h;
            let x0 = patch_col as usize * cfg.patch_w;
            for out_channel in 0..cfg.out_channels {
                let mut acc = bias[out_channel];
                for c in 0..cfg.in_channels {
                    for dt in 0..cfg.tubelet_size {
                        let t = t0 + dt;
                        for py in 0..cfg.patch_h {
                            let y = y0 + py;
                            for px in 0..cfg.patch_w {
                                let x = x0 + px;
                                let input_idx = (((batch * cfg.in_channels + c) * cfg.frames + t)
                                    * cfg.height
                                    + y)
                                    * cfg.width
                                    + x;
                                let weight_idx = ((((out_channel * cfg.in_channels + c)
                                    * cfg.tubelet_size
                                    + dt)
                                    * cfg.patch_h
                                    + py)
                                    * cfg.patch_w)
                                    + px;
                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                dense_ref.push(acc);
            }
        }

        assert_eq!(dense_ref.len(), sparse.len());
        for (idx, (lhs, rhs)) in dense_ref.iter().zip(sparse.iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            assert!(
                diff <= 2.5e-5,
                "patchify mismatch at idx={idx}: dense={lhs} sparse={rhs} diff={diff}"
            );
        }
    }

    #[test]
    fn sparse_patchify3d_handles_empty_coords() {
        let cfg = SparsePatchify3dConfig {
            in_channels: 1,
            out_channels: 2,
            frames: 2,
            height: 4,
            width: 4,
            tubelet_size: 2,
            patch_h: 2,
            patch_w: 2,
        };
        let weights = SparsePatchify3dWeights {
            weight: &[1.0; 16],
            bias: &[0.0, 1.0],
        };
        let out = sparse_patchify3d_forward_flex(&cfg, weights, &[], &[0.0; 32]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn sparse_patchify3d_rejects_out_of_bounds_coord() {
        let cfg = SparsePatchify3dConfig {
            in_channels: 1,
            out_channels: 1,
            frames: 2,
            height: 4,
            width: 4,
            tubelet_size: 2,
            patch_h: 2,
            patch_w: 2,
        };
        let weights = SparsePatchify3dWeights {
            weight: &[1.0; 8],
            bias: &[0.0],
        };
        let err =
            sparse_patchify3d_forward_flex(&cfg, weights, &[[0, 0, 2, 0]], &[0.0; 32]).unwrap_err();
        assert!(
            err.contains("patch_row out of bounds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn sparse_patchify3d_weight_pack_uses_patch_order() {
        let cfg = SparsePatchify3dConfig {
            in_channels: 2,
            out_channels: 3,
            frames: 2,
            height: 2,
            width: 2,
            tubelet_size: 2,
            patch_h: 1,
            patch_w: 2,
        };
        let weight: Vec<f32> =
            (0..cfg.out_channels * cfg.in_channels * cfg.tubelet_size * cfg.patch_h * cfg.patch_w)
                .map(|idx| idx as f32)
                .collect();
        let packed = pack_sparse_patchify3d_weight(&cfg, &weight).unwrap();
        let inner = cfg.in_channels * cfg.tubelet_size * cfg.patch_h * cfg.patch_w;
        assert_eq!(packed.len(), inner * cfg.out_channels);
        for out_channel in 0..cfg.out_channels {
            let mut k_idx = 0usize;
            for c in 0..cfg.in_channels {
                for dt in 0..cfg.tubelet_size {
                    for py in 0..cfg.patch_h {
                        for px in 0..cfg.patch_w {
                            let src = ((((out_channel * cfg.in_channels + c) * cfg.tubelet_size
                                + dt)
                                * cfg.patch_h
                                + py)
                                * cfg.patch_w)
                                + px;
                            assert_eq!(packed[k_idx * cfg.out_channels + out_channel], weight[src]);
                            k_idx += 1;
                        }
                    }
                }
            }
        }
    }
}

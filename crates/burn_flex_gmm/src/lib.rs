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

#[derive(Clone, Debug)]
struct KernelLayout {
    offsets: Vec<[i32; 3]>,
    rows: usize,
}

#[cfg(feature = "wgpu-kernel")]
pub mod wgpu;

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

    let kernel = kernel_layout(config);
    let mut output = vec![0.0f32; rows * config.out_channels];
    for row_idx in 0..rows {
        let base = row_idx * config.out_channels;
        output[base..base + config.out_channels].copy_from_slice(weights.bias);
    }

    let neighbor_rows = build_neighbor_rows(config, coords)?;

    let k_in = kernel
        .rows
        .checked_mul(config.in_channels_per_group)
        .ok_or_else(|| "k dimension overflow in sparse_subm_conv_forward_flex".to_string())?;
    let m = rows;
    let n = config.out_channels_per_group;
    for group in 0..config.groups {
        let mut gathered = vec![0.0f32; m * k_in];
        let in_group_base = group * config.in_channels_per_group;
        for out_row in 0..m {
            for kernel_idx in 0..kernel.rows {
                let in_row = neighbor_rows[out_row * kernel.rows + kernel_idx];
                if in_row == INVALID_NEIGHBOR_ROW {
                    continue;
                }
                let in_row = in_row as usize;
                let src_base = in_row * config.in_channels + in_group_base;
                let dst_base = out_row * k_in + kernel_idx * config.in_channels_per_group;
                gathered[dst_base..dst_base + config.in_channels_per_group]
                    .copy_from_slice(&input[src_base..src_base + config.in_channels_per_group]);
            }
        }

        let mut packed_weight = vec![0.0f32; k_in * n];
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
                            packed_weight[k_col * n + out_local] = weights.weight[src_idx];
                        }
                    }
                }
            }
        }

        let out_group_base = group * config.out_channels_per_group;
        let mut out_group = vec![0.0f32; m * n];
        for row in 0..m {
            let src_base = row * config.out_channels + out_group_base;
            let dst_base = row * n;
            out_group[dst_base..dst_base + n].copy_from_slice(&output[src_base..src_base + n]);
        }

        // out_group[m, n] += gathered[m, k] @ packed_weight[k, n]
        unsafe {
            matrixmultiply::sgemm(
                m,
                k_in,
                n,
                1.0,
                gathered.as_ptr(),
                k_in as isize,
                1,
                packed_weight.as_ptr(),
                n as isize,
                1,
                1.0,
                out_group.as_mut_ptr(),
                n as isize,
                1,
            );
        }

        for row in 0..m {
            let src_base = row * n;
            let dst_base = row * config.out_channels + out_group_base;
            output[dst_base..dst_base + n].copy_from_slice(&out_group[src_base..src_base + n]);
        }
    }

    Ok(output)
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
    let expected_weight = config
        .out_channels
        .checked_mul(config.kernel_d)
        .and_then(|v| v.checked_mul(config.kernel_h))
        .and_then(|v| v.checked_mul(config.kernel_w))
        .and_then(|v| v.checked_mul(config.in_channels_per_group))
        .ok_or_else(|| "sparse conv weight size overflow".to_string())?;
    if weights.weight.len() != expected_weight {
        return Err(format!(
            "sparse conv weight len mismatch: got {} expected {}",
            weights.weight.len(),
            expected_weight
        ));
    }
    Ok(())
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
        SparseSubmConvConfig, SparseSubmConvWeights, sparse_subm_conv_forward_flex,
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
}

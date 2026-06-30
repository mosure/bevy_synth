use std::{
    env,
    sync::{Mutex, MutexGuard},
};

use burn::tensor::{Int, Tensor, TensorData};

use crate::{SparseSubmConvConfig, SparseSubmConvWeights, sparse_subm_conv_forward_flex};

use super::{
    DefaultWgpuBackend, NeighborDeviceAlgoPreference, SparseConvKernelVariant,
    SparseWgpuForwardConfig, SparseWgpuKernelVariant, build_neighbor_rows_tensor_device_scan,
    clear_neighbor_rows_tensor_cache, dense_trilinear_sample_attrs_wgpu,
    layer_norm_affine_forward_wgpu, layer_norm_affine_silu_forward_wgpu,
    layer_norm_modulated_forward_wgpu, layer_norm_row_stats_debug_wgpu, linear_skinny_forward_wgpu,
    multihead_qk_rms_norm_rope_from_qkv_coords_wgpu,
    multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu, multihead_rms_norm_forward_wgpu,
    multihead_rms_norm_module_forward_wgpu, multihead_rms_norm_rope_from_coords_wgpu,
    neighbor_rows_build_stats, neighbor_rows_tensor_from_coords,
    neighbor_rows_tensor_from_coords_tensor, neighbor_rows_tensor_from_coords_with_algo,
    reset_neighbor_rows_build_stats, reset_sparse_wgpu_kernel_stats,
    resolve_sparse_wgpu_forward_config, resolve_sparse_wgpu_forward_config_internal,
    rope_rotate_pairs_from_coords_wgpu, rope_rotate_pairs_from_phase_wgpu, rope_rotate_pairs_wgpu,
    sparse_subm_conv_forward_wgpu, sparse_subm_conv_forward_wgpu_im2col_matmul,
    sparse_subm_conv_forward_wgpu_im2col_matmul_fast_f16,
    sparse_subm_conv_forward_wgpu_with_config, sparse_wgpu_kernel_stats,
};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock_guard() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn wgpu_test_device() -> Option<burn_wgpu::WgpuDevice> {
    if env::var("BURN_WGPU_CORRECTNESS").ok().as_deref() != Some("1") {
        eprintln!(
            "skipping WGPU correctness test; set BURN_WGPU_CORRECTNESS=1 to run GPU kernel parity"
        );
        return None;
    }
    Some(burn_wgpu::WgpuDevice::default())
}

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

fn dense_sample_reference(
    position: [f32; 3],
    occupancy: &[i32],
    attrs: &[[f32; 6]],
    spatial: [usize; 3],
) -> Option<[f32; 6]> {
    let map_axis = |value: f32, dim: usize| -> f32 {
        let dim = dim.max(1) as f32;
        ((value + 0.5) * dim).clamp(0.0, dim)
    };
    let coord = [
        map_axis(position[0], spatial[0]),
        map_axis(position[1], spatial[1]),
        map_axis(position[2], spatial[2]),
    ];
    let base = [
        (coord[0] - 0.5).floor() as i32,
        (coord[1] - 0.5).floor() as i32,
        (coord[2] - 0.5).floor() as i32,
    ];

    let max_x = spatial[0].saturating_sub(1) as i32;
    let max_y = spatial[1].saturating_sub(1) as i32;
    let max_z = spatial[2].saturating_sub(1) as i32;
    let x0 = base[0];
    let y0 = base[1];
    let z0 = base[2];
    let x1 = base[0] + 1;
    let y1 = base[1] + 1;
    let z1 = base[2] + 1;
    let weight_axis =
        |query: f32, cell: i32| -> f32 { (1.0 - (query - cell as f32 - 0.5).abs()).max(0.0) };
    let wx0 = weight_axis(coord[0], x0);
    let wy0 = weight_axis(coord[1], y0);
    let wz0 = weight_axis(coord[2], z0);
    let wx1 = weight_axis(coord[0], x1);
    let wy1 = weight_axis(coord[1], y1);
    let wz1 = weight_axis(coord[2], z1);
    let stride_x = spatial[0];
    let stride_xy = spatial[0].saturating_mul(spatial[1]);
    let idx = |x: usize, y: usize, z: usize| -> usize { z * stride_xy + y * stride_x + x };

    let mut accum = [0.0f32; 6];
    let mut weight_sum = 0.0f32;
    let mut sample_corner = |x: i32, y: i32, z: i32, weight: f32| {
        if weight <= 0.0 {
            return;
        }
        if x < 0 || x > max_x || y < 0 || y > max_y || z < 0 || z > max_z {
            return;
        }
        let x = x as usize;
        let y = y as usize;
        let z = z as usize;
        let linear = idx(x, y, z);
        if occupancy[linear] == 0 {
            return;
        }
        for ch in 0..6 {
            accum[ch] += attrs[linear][ch] * weight;
        }
        weight_sum += weight;
    };
    sample_corner(x0, y0, z0, wx0 * wy0 * wz0);
    sample_corner(x1, y0, z0, wx1 * wy0 * wz0);
    sample_corner(x0, y1, z0, wx0 * wy1 * wz0);
    sample_corner(x1, y1, z0, wx1 * wy1 * wz0);
    sample_corner(x0, y0, z1, wx0 * wy0 * wz1);
    sample_corner(x1, y0, z1, wx1 * wy0 * wz1);
    sample_corner(x0, y1, z1, wx0 * wy1 * wz1);
    sample_corner(x1, y1, z1, wx1 * wy1 * wz1);
    if weight_sum <= 1.0e-8 {
        return None;
    }
    let inv = 1.0 / weight_sum;
    for value in &mut accum {
        *value *= inv;
    }
    Some(accum)
}

#[test]
fn dense_trilinear_sample_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let spatial = [16usize, 16usize, 16usize];
    let cells = spatial[0] * spatial[1] * spatial[2];
    let mut occupancy = vec![0i32; cells];
    let mut attrs = vec![[0.0f32; 6]; cells];
    let stride_x = spatial[0];
    let stride_xy = spatial[0] * spatial[1];
    let linear = |x: usize, y: usize, z: usize| -> usize { z * stride_xy + y * stride_x + x };
    for z in 4..12 {
        for y in 4..12 {
            for x in 4..12 {
                let idx = linear(x, y, z);
                occupancy[idx] = 1;
                attrs[idx] = [
                    x as f32 / 15.0,
                    y as f32 / 15.0,
                    z as f32 / 15.0,
                    0.1 + x as f32 / 30.0,
                    0.2 + y as f32 / 30.0,
                    1.0,
                ];
            }
        }
    }

    let positions = vec![
        [0.0, 0.0, 0.0],
        [(7.0 / 16.0) - 0.5, (7.0 / 16.0) - 0.5, (7.0 / 16.0) - 0.5],
        [(7.5 / 16.0) - 0.5, (8.0 / 16.0) - 0.5, (8.5 / 16.0) - 0.5],
        [0.45, -0.45, 0.45],
    ];
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let mut positions_flat = Vec::with_capacity(positions.len() * 3);
    for pos in &positions {
        positions_flat.extend_from_slice(pos);
    }
    let mut attrs_flat = Vec::with_capacity(cells * 6);
    for row in &attrs {
        attrs_flat.extend_from_slice(row);
    }
    let positions_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(positions_flat, [positions.len(), 3]),
        &device,
    );
    let occupancy_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(occupancy.clone(), [cells]),
        &device,
    );
    let attrs_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(attrs_flat, [cells, 6]),
        &device,
    );
    let sampled_t = dense_trilinear_sample_attrs_wgpu(positions_t, occupancy_t, attrs_t, spatial)
        .expect("kernel sample");
    let sampled = sampled_t.to_data().as_slice::<f32>().expect("f32").to_vec();
    assert_eq!(sampled.len(), positions.len() * 7);
    for (row, position) in positions.iter().enumerate() {
        let expected =
            dense_sample_reference(*position, occupancy.as_slice(), attrs.as_slice(), spatial);
        let base = row * 7;
        let support = sampled[base + 6];
        match expected {
            Some(values) => {
                assert!(
                    support > 1.0e-8,
                    "expected supported sample at row {row}, got support={support}"
                );
                for ch in 0..6 {
                    let diff = (sampled[base + ch] - values[ch]).abs();
                    assert!(
                        diff <= 1.0e-4,
                        "dense sample mismatch row={row} ch={ch}: got={} expected={} diff={diff}",
                        sampled[base + ch],
                        values[ch]
                    );
                }
            }
            None => {
                assert!(
                    support <= 1.0e-8,
                    "expected unsupported sample at row {row}, got support={support}"
                );
            }
        }
    }
}

#[test]
fn rope_rotate_pairs_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 5usize;
    let heads = 3usize;
    let head_dim = 16usize;
    let pairs = head_dim / 2;
    let rows = batch * tokens * heads;

    let mut rng = Lcg::new(0xC0FFEEu64);
    let mut x = vec![0.0f32; rows * head_dim];
    for value in &mut x {
        *value = rng.next_f32();
    }
    let mut phase = vec![0.0f32; tokens * pairs];
    for (idx, value) in phase.iter_mut().enumerate() {
        *value = idx as f32 * 0.013 + 0.37;
    }
    let cos = phase.iter().map(|value| value.cos()).collect::<Vec<_>>();
    let sin = phase.iter().map(|value| value.sin()).collect::<Vec<_>>();

    let x_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(x.as_slice(), &device)
        .reshape([batch, tokens, heads, head_dim]);
    let cos_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(cos.as_slice(), &device)
        .reshape([1, tokens, 1, pairs]);
    let sin_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(sin.as_slice(), &device)
        .reshape([1, tokens, 1, pairs]);

    let rotated = rope_rotate_pairs_wgpu(x_t, cos_t, sin_t).expect("rope kernel output");
    let rotated = rotated
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut max_abs = 0.0f32;
    for b in 0..batch {
        for t in 0..tokens {
            for h in 0..heads {
                let row = (b * tokens + t) * heads + h;
                for p in 0..pairs {
                    let base = row * head_dim + p * 2;
                    let x_even = x[base];
                    let x_odd = x[base + 1];
                    let c = cos[t * pairs + p];
                    let s = sin[t * pairs + p];
                    let ref_even = x_even * c - x_odd * s;
                    let ref_odd = x_even * s + x_odd * c;
                    max_abs = max_abs.max((rotated[base] - ref_even).abs());
                    max_abs = max_abs.max((rotated[base + 1] - ref_odd).abs());
                }
            }
        }
    }
    assert!(
        max_abs <= 1.0e-5,
        "rope rotate kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn rope_rotate_pairs_phase_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 5usize;
    let heads = 3usize;
    let head_dim = 16usize;
    let pairs = head_dim / 2;
    let rows = batch * tokens * heads;

    let mut rng = Lcg::new(0x5EED_BAADu64);
    let mut x = vec![0.0f32; rows * head_dim];
    for value in &mut x {
        *value = rng.next_f32();
    }
    let mut phase = vec![0.0f32; tokens * pairs];
    for (idx, value) in phase.iter_mut().enumerate() {
        *value = idx as f32 * 0.017 + 0.11;
    }

    let x_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(x.as_slice(), &device)
        .reshape([batch, tokens, heads, head_dim]);
    let phase_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(phase.as_slice(), &device)
        .reshape([tokens, pairs]);

    let rotated = rope_rotate_pairs_from_phase_wgpu(x_t, phase_t).expect("rope phase output");
    let rotated = rotated
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut max_abs = 0.0f32;
    for b in 0..batch {
        for t in 0..tokens {
            for h in 0..heads {
                let row = (b * tokens + t) * heads + h;
                for p in 0..pairs {
                    let base = row * head_dim + p * 2;
                    let x_even = x[base];
                    let x_odd = x[base + 1];
                    let phase_v = phase[t * pairs + p];
                    let c = phase_v.cos();
                    let s = phase_v.sin();
                    let ref_even = x_even * c - x_odd * s;
                    let ref_odd = x_even * s + x_odd * c;
                    max_abs = max_abs.max((rotated[base] - ref_even).abs());
                    max_abs = max_abs.max((rotated[base + 1] - ref_odd).abs());
                }
            }
        }
    }
    assert!(
        max_abs <= 1.0e-5,
        "rope rotate phase kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn rope_rotate_pairs_coords_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 7usize;
    let heads = 3usize;
    let head_dim = 16usize;
    let pairs = head_dim / 2;
    let rows = batch * tokens * heads;
    let rope_freq = [1.0f32, 10_000.0f32];

    let mut rng = Lcg::new(0xA11CEu64);
    let mut x = vec![0.0f32; rows * head_dim];
    for value in &mut x {
        *value = rng.next_f32();
    }
    let mut coords = vec![0i32; tokens * 3];
    for token in 0..tokens {
        coords[token * 3] = token as i32;
        coords[token * 3 + 1] = (token as i32) * 2 - 3;
        coords[token * 3 + 2] = (token as i32) * 3 + 1;
    }

    let x_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(x.as_slice(), &device)
        .reshape([batch, tokens, heads, head_dim]);
    let coords_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords.clone(), [tokens * 3]),
        &device,
    )
    .reshape([tokens, 3]);

    let rotated =
        rope_rotate_pairs_from_coords_wgpu(x_t, coords_t, rope_freq).expect("rope coords output");
    let rotated = rotated
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let freq_dim = (pairs / 3).max(1);
    let mut max_abs = 0.0f32;
    for b in 0..batch {
        for t in 0..tokens {
            for h in 0..heads {
                let row = (b * tokens + t) * heads + h;
                for p in 0..pairs {
                    let (axis, freq_idx) = if p < freq_dim {
                        (0usize, p)
                    } else if p < freq_dim * 2 {
                        (1usize, p - freq_dim)
                    } else if p < freq_dim * 3 {
                        (2usize, p - freq_dim * 2)
                    } else {
                        (usize::MAX, 0usize)
                    };
                    let phase = if axis == usize::MAX {
                        0.0
                    } else {
                        let exp = freq_idx as f32 / freq_dim as f32;
                        let freq = rope_freq[0] / rope_freq[1].powf(exp);
                        coords[t * 3 + axis] as f32 * freq
                    };
                    let c = phase.cos();
                    let s = phase.sin();
                    let base = row * head_dim + p * 2;
                    let x_even = x[base];
                    let x_odd = x[base + 1];
                    let ref_even = x_even * c - x_odd * s;
                    let ref_odd = x_even * s + x_odd * c;
                    max_abs = max_abs.max((rotated[base] - ref_even).abs());
                    max_abs = max_abs.max((rotated[base + 1] - ref_odd).abs());
                }
            }
        }
    }
    assert!(
        max_abs <= 1.0e-5,
        "rope rotate coords kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn linear_skinny_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let rows = 3072usize;
    let in_channels = 64usize;
    let out_channels = 7usize;
    let mut rng = Lcg::new(0x51A11CEu64);
    let mut input = vec![0.0f32; rows * in_channels];
    let mut weight = vec![0.0f32; out_channels * in_channels];
    let mut bias = vec![0.0f32; out_channels];
    for value in &mut input {
        *value = rng.next_f32();
    }
    for value in &mut weight {
        *value = rng.next_f32();
    }
    for value in &mut bias {
        *value = rng.next_f32();
    }

    let input_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(input.clone(), [rows, in_channels]),
        &device,
    );
    let weight_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(weight.clone(), [out_channels, in_channels]),
        &device,
    );
    let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(bias.clone(), [out_channels]),
        &device,
    );
    let output =
        linear_skinny_forward_wgpu(input_t, weight_t, bias_t).expect("skinny linear kernel output");
    let output = output.to_data().as_slice::<f32>().expect("f32").to_vec();

    let mut max_abs = 0.0f32;
    for row in 0..rows {
        for out_idx in 0..out_channels {
            let mut expected = bias[out_idx];
            for in_idx in 0..in_channels {
                expected +=
                    input[row * in_channels + in_idx] * weight[out_idx * in_channels + in_idx];
            }
            let actual = output[row * out_channels + out_idx];
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    assert!(
        max_abs <= 1.0e-4,
        "skinny linear kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn layer_norm_affine_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let rows = 1024usize;
    let channels = 64usize;
    let eps = 1.0e-6f32;
    let mut rng = Lcg::new(0x1A2B3C4Du64);
    let mut input = vec![0.0f32; rows * channels];
    let mut weight = vec![0.0f32; channels];
    let mut bias = vec![0.0f32; channels];
    for value in &mut input {
        *value = rng.next_f32();
    }
    for value in &mut weight {
        *value = rng.next_f32();
    }
    for value in &mut bias {
        *value = rng.next_f32();
    }

    let input_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(input.clone(), [rows, channels]),
        &device,
    );
    let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(weight.clone(), [channels]),
        &device,
    );
    let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(bias.clone(), [channels]),
        &device,
    );
    let output = layer_norm_affine_forward_wgpu(input_t, weight_t, bias_t, eps)
        .expect("layer norm kernel output");
    let output = output.to_data().as_slice::<f32>().expect("f32").to_vec();

    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let base = row * channels;
        let mut mean = 0.0f32;
        for ch in 0..channels {
            mean += input[base + ch];
        }
        mean /= channels as f32;
        let mut var = 0.0f32;
        for ch in 0..channels {
            let centered = input[base + ch] - mean;
            var += centered * centered;
        }
        var /= channels as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ch in 0..channels {
            let expected = (input[base + ch] - mean) * inv_std * weight[ch] + bias[ch];
            let actual = output[base + ch];
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    assert!(
        max_abs <= 2.0e-4,
        "layer norm affine kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn layer_norm_affine_partial_stats_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let rows = 1024usize;
    let channels = 1536usize;
    let eps = 1.0e-6f32;
    let mut rng = Lcg::new(0x1A2B_1536u64);
    let mut input = vec![0.0f32; rows * channels];
    let mut weight = vec![0.0f32; channels];
    let mut bias = vec![0.0f32; channels];
    for value in &mut input {
        *value = rng.next_f32();
    }
    for value in &mut weight {
        *value = rng.next_f32() * 0.5;
    }
    for value in &mut bias {
        *value = rng.next_f32() * 0.25;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(input.clone(), [rows, channels]),
        &device,
    );
    let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(weight.clone(), [channels]),
        &device,
    );
    let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(bias.clone(), [channels]),
        &device,
    );
    let stats_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(input.clone(), [rows, channels]),
        &device,
    );
    let stats = layer_norm_row_stats_debug_wgpu(stats_t)
        .expect("layer norm partial stats debug output")
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let output = layer_norm_affine_forward_wgpu(input_t, weight_t, bias_t, eps)
        .expect("layer norm partial stats kernel output");
    let output = output.to_data().as_slice::<f32>().expect("f32").to_vec();

    let mut max_mean_abs = 0.0f32;
    let mut max_var_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let base = row * channels;
        let mut mean = 0.0f32;
        for ch in 0..channels {
            mean += input[base + ch];
        }
        mean /= channels as f32;
        let mut var = 0.0f32;
        for ch in 0..channels {
            let centered = input[base + ch] - mean;
            var += centered * centered;
        }
        var /= channels as f32;
        max_mean_abs = max_mean_abs.max((stats[row * 2] - mean).abs());
        max_var_abs = max_var_abs.max((stats[row * 2 + 1] - var).abs());
        let inv_std = 1.0 / (var + eps).sqrt();
        for ch in 0..channels {
            let expected = (input[base + ch] - mean) * inv_std * weight[ch] + bias[ch];
            let actual = output[base + ch];
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    assert!(
        max_mean_abs <= 1.0e-5 && max_var_abs <= 1.0e-5,
        "layer norm affine partial stats drift too high: max_mean_abs={max_mean_abs:.6e} max_var_abs={max_var_abs:.6e}"
    );
    assert!(
        max_abs <= 5.0e-4,
        "layer norm affine partial stats kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn layer_norm_modulated_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 3usize;
    let tokens = 19usize;
    let channels = 64usize;
    let eps = 1.0e-6f32;
    let mut rng = Lcg::new(0xADAD_A11Cu64);
    let mut input = vec![0.0f32; batch * tokens * channels];
    let mut scale = vec![0.0f32; batch * channels];
    let mut shift = vec![0.0f32; batch * channels];
    for value in &mut input {
        *value = rng.next_f32();
    }
    for value in &mut scale {
        *value = rng.next_f32() * 0.25;
    }
    for value in &mut shift {
        *value = rng.next_f32() * 0.5;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(input.clone(), [batch, tokens, channels]),
        &device,
    );
    let scale_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(scale.clone(), [batch, 1, channels]),
        &device,
    );
    let shift_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(shift.clone(), [batch, 1, channels]),
        &device,
    );
    let output = layer_norm_modulated_forward_wgpu(input_t, scale_t, shift_t, eps)
        .expect("layer norm modulation kernel output");
    let output = output.to_data().as_slice::<f32>().expect("f32").to_vec();

    let mut max_abs = 0.0f32;
    for b in 0..batch {
        for t in 0..tokens {
            let row = b * tokens + t;
            let base = row * channels;
            let mut mean = 0.0f32;
            for ch in 0..channels {
                mean += input[base + ch];
            }
            mean /= channels as f32;
            let mut var = 0.0f32;
            for ch in 0..channels {
                let centered = input[base + ch] - mean;
                var += centered * centered;
            }
            var /= channels as f32;
            let inv_std = (var + eps).sqrt().recip();
            for ch in 0..channels {
                let mod_idx = b * channels + ch;
                let expected =
                    (input[base + ch] - mean) * inv_std * (scale[mod_idx] + 1.0) + shift[mod_idx];
                let actual = output[base + ch];
                max_abs = max_abs.max((actual - expected).abs());
            }
        }
    }
    assert!(
        max_abs <= 2.0e-4,
        "layer norm modulation kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn layer_norm_modulated_f16_partial_stats_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 33usize;
    let channels = 1536usize;
    let eps = 1.0e-6f32;
    let mut rng = Lcg::new(0xF16_ADA11Cu64);
    let mut input = vec![0.0f32; batch * tokens * channels];
    let mut scale = vec![0.0f32; batch * channels];
    let mut shift = vec![0.0f32; batch * channels];
    for value in &mut input {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut scale {
        *value = rng.next_f32() * 0.25;
    }
    for value in &mut shift {
        *value = rng.next_f32() * 0.5;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(input.clone(), [batch, tokens, channels]),
        &device,
    )
    .cast(burn::tensor::FloatDType::F16);
    let scale_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(scale.clone(), [batch, 1, channels]),
        &device,
    )
    .cast(burn::tensor::FloatDType::F16);
    let shift_t = Tensor::<DefaultWgpuBackend, 3>::from_data(
        TensorData::new(shift.clone(), [batch, 1, channels]),
        &device,
    )
    .cast(burn::tensor::FloatDType::F16);
    let output = layer_norm_modulated_forward_wgpu(input_t, scale_t, shift_t, eps)
        .expect("layer norm modulation f16 kernel output");
    assert_eq!(
        burn::tensor::FloatDType::from(output.dtype()),
        burn::tensor::FloatDType::F16
    );
    let output = output.to_data().convert::<f32>().to_vec::<f32>().unwrap();

    let mut max_abs = 0.0f32;
    let mut mean_abs = 0.0f64;
    for b in 0..batch {
        for t in 0..tokens {
            let row = b * tokens + t;
            let base = row * channels;
            let mut mean = 0.0f32;
            for ch in 0..channels {
                mean += half::f16::from_f32(input[base + ch]).to_f32();
            }
            mean /= channels as f32;
            let mut var = 0.0f32;
            for ch in 0..channels {
                let centered = half::f16::from_f32(input[base + ch]).to_f32() - mean;
                var += centered * centered;
            }
            var /= channels as f32;
            let inv_std = (var + eps).sqrt().recip();
            for ch in 0..channels {
                let mod_idx = b * channels + ch;
                let input_v = half::f16::from_f32(input[base + ch]).to_f32();
                let scale_v = half::f16::from_f32(scale[mod_idx]).to_f32();
                let shift_v = half::f16::from_f32(shift[mod_idx]).to_f32();
                let expected =
                    half::f16::from_f32((input_v - mean) * inv_std * (scale_v + 1.0) + shift_v)
                        .to_f32();
                let diff = (output[base + ch] - expected).abs();
                max_abs = max_abs.max(diff);
                mean_abs += f64::from(diff);
            }
        }
    }
    mean_abs /= output.len() as f64;
    assert!(
        max_abs <= 3.0e-3 && mean_abs <= 3.0e-4,
        "layer norm modulation f16 kernel drift too high: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e}"
    );
}

#[test]
fn multihead_rms_norm_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 17usize;
    let heads = 4usize;
    let head_dim = 64usize;
    let rows = batch * tokens * heads;
    let eps = 1.0e-12f32;
    let scale = (head_dim as f32).sqrt();
    let mut rng = Lcg::new(0xA11C_E123u64);
    let mut input = vec![0.0f32; rows * head_dim];
    let mut gamma = vec![0.0f32; heads * head_dim];
    for value in &mut input {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(input.clone(), [rows * head_dim]),
        &device,
    )
    .reshape([batch, tokens, heads, head_dim]);
    let gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let output = multihead_rms_norm_forward_wgpu(input_t, gamma_t, scale, eps)
        .expect("multihead rms norm output");
    let output = output
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let base = row * head_dim;
        let head = row % heads;
        let mut sq_sum = 0.0f32;
        for ch in 0..head_dim {
            let value = input[base + ch];
            sq_sum += value * value;
        }
        let inv_rms = (sq_sum + eps).sqrt().recip();
        for ch in 0..head_dim {
            let expected = input[base + ch] * inv_rms * scale * gamma[head * head_dim + ch];
            let actual = output[base + ch];
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    assert!(
        max_abs <= 1.0e-5,
        "multihead rms norm kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn multihead_rms_norm_module_kernel_matches_permuted_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 13usize;
    let heads = 3usize;
    let head_dim = 32usize;
    let rows = batch * tokens * heads;
    let eps = 1.0e-12f32;
    let scale = (head_dim as f32).sqrt();
    let mut rng = Lcg::new(0xA11C_E124u64);
    let mut input = vec![0.0f32; rows * head_dim];
    let mut gamma = vec![0.0f32; heads * head_dim];
    for value in &mut input {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(input.clone(), [rows * head_dim]),
        &device,
    )
    .reshape([batch, tokens, heads, head_dim]);
    let gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let reference = multihead_rms_norm_forward_wgpu(input_t.clone(), gamma_t.clone(), scale, eps)
        .expect("multihead rms norm output")
        .permute([0, 2, 1, 3]);
    let output = multihead_rms_norm_module_forward_wgpu(input_t, gamma_t, scale, eps)
        .expect("multihead rms norm module output");
    let reference = reference
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap();
    let output = output.to_data().convert::<f32>().to_vec::<f32>().unwrap();

    let mut max_abs = 0.0f32;
    for (actual, expected) in output.iter().copied().zip(reference.iter().copied()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    assert!(
        max_abs <= 1.0e-5,
        "multihead rms norm module kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn multihead_rms_norm_rope_coords_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 11usize;
    let heads = 3usize;
    let head_dim = 18usize;
    let pairs = head_dim / 2;
    let rows = batch * tokens * heads;
    let eps = 1.0e-12f32;
    let scale = (head_dim as f32).sqrt();
    let rope_freq = [1.0f32, 10_000.0f32];
    let mut rng = Lcg::new(0xA11C_E456u64);
    let mut input = vec![0.0f32; rows * head_dim];
    let mut gamma = vec![0.0f32; heads * head_dim];
    let mut coords = vec![0i32; tokens * 3];
    for value in &mut input {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }
    for token in 0..tokens {
        coords[token * 3] = token as i32 - 3;
        coords[token * 3 + 1] = (token as i32) * 2 + 1;
        coords[token * 3 + 2] = (token as i32) * 3 - 2;
    }

    let input_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(input.clone(), [rows * head_dim]),
        &device,
    )
    .reshape([batch, tokens, heads, head_dim]);
    let gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let coords_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords.clone(), [tokens * 3]),
        &device,
    )
    .reshape([tokens, 3]);
    let output =
        multihead_rms_norm_rope_from_coords_wgpu(input_t, gamma_t, coords_t, rope_freq, scale, eps)
            .expect("multihead rms norm rope output");
    let output = output
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let freq_dim = (pairs / 3).max(1);
    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let base = row * head_dim;
        let token = (row / heads) % tokens;
        let head = row % heads;
        let mut sq_sum = 0.0f32;
        for ch in 0..head_dim {
            let value = input[base + ch];
            sq_sum += value * value;
        }
        let inv_rms = (sq_sum + eps).sqrt().recip();
        for pair in 0..pairs {
            let even_ch = pair * 2;
            let odd_ch = even_ch + 1;
            let even = input[base + even_ch] * inv_rms * scale * gamma[head * head_dim + even_ch];
            let odd = input[base + odd_ch] * inv_rms * scale * gamma[head * head_dim + odd_ch];
            let (axis, freq_idx) = if pair < freq_dim {
                (0usize, pair)
            } else if pair < freq_dim * 2 {
                (1usize, pair - freq_dim)
            } else if pair < freq_dim * 3 {
                (2usize, pair - freq_dim * 2)
            } else {
                (usize::MAX, 0usize)
            };
            let phase = if axis == usize::MAX {
                0.0
            } else {
                let exp = freq_idx as f32 / freq_dim as f32;
                let freq = rope_freq[0] / rope_freq[1].powf(exp);
                coords[token * 3 + axis] as f32 * freq
            };
            let c = phase.cos();
            let s = phase.sin();
            let expected_even = even * c - odd * s;
            let expected_odd = even * s + odd * c;
            max_abs = max_abs.max((output[base + even_ch] - expected_even).abs());
            max_abs = max_abs.max((output[base + odd_ch] - expected_odd).abs());
        }
    }
    assert!(
        max_abs <= 1.0e-5,
        "multihead rms norm rope kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn multihead_qk_rms_norm_rope_qkv_kernel_matches_separate_kernels() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 13usize;
    let heads = 3usize;
    let head_dim = 18usize;
    let rows = batch * tokens * heads;
    let eps = 1.0e-12f32;
    let scale = (head_dim as f32).sqrt();
    let rope_freq = [1.0f32, 10_000.0f32];
    let mut rng = Lcg::new(0xA11C_E789u64);
    let mut qkv = vec![0.0f32; batch * tokens * 3 * heads * head_dim];
    let mut q_gamma = vec![0.0f32; heads * head_dim];
    let mut k_gamma = vec![0.0f32; heads * head_dim];
    let mut coords = vec![0i32; tokens * 3];
    for value in &mut qkv {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut q_gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }
    for value in &mut k_gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }
    for token in 0..tokens {
        coords[token * 3] = token as i32 - 5;
        coords[token * 3 + 1] = (token as i32) * 2 - 1;
        coords[token * 3 + 2] = (token as i32) * 3 + 2;
    }

    let qkv_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(qkv.clone(), [qkv.len()]),
        &device,
    )
    .reshape([batch, tokens, 3, heads, head_dim]);
    let q_t = qkv_t
        .clone()
        .slice([0..batch, 0..tokens, 0..1, 0..heads, 0..head_dim])
        .reshape([batch, tokens, heads, head_dim]);
    let k_t = qkv_t
        .clone()
        .slice([0..batch, 0..tokens, 1..2, 0..heads, 0..head_dim])
        .reshape([batch, tokens, heads, head_dim]);
    let q_gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(q_gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let k_gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(k_gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let coords_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords.clone(), [tokens * 3]),
        &device,
    )
    .reshape([tokens, 3]);

    let q_reference = multihead_rms_norm_rope_from_coords_wgpu(
        q_t,
        q_gamma_t.clone(),
        coords_t.clone(),
        rope_freq,
        scale,
        eps,
    )
    .expect("q separate rms norm rope output");
    let k_reference = multihead_rms_norm_rope_from_coords_wgpu(
        k_t,
        k_gamma_t.clone(),
        coords_t.clone(),
        rope_freq,
        scale,
        eps,
    )
    .expect("k separate rms norm rope output");
    let (q_fused, k_fused) = multihead_qk_rms_norm_rope_from_qkv_coords_wgpu(
        qkv_t, q_gamma_t, k_gamma_t, coords_t, rope_freq, scale, eps,
    )
    .expect("fused qk rms norm rope output");

    let q_reference = q_reference
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let k_reference = k_reference
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let q_fused = q_fused
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let k_fused = k_fused
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut max_abs = 0.0f32;
    for (actual, expected) in q_fused.iter().zip(q_reference.iter()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    for (actual, expected) in k_fused.iter().zip(k_reference.iter()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    assert!(
        max_abs <= 1.0e-5,
        "fused qk rms norm rope kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn multihead_qkv_module_rms_norm_rope_qkv_kernel_matches_module_layout_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let batch = 2usize;
    let tokens = 13usize;
    let heads = 3usize;
    let head_dim = 18usize;
    let rows = batch * heads * tokens;
    let eps = 1.0e-12f32;
    let scale = (head_dim as f32).sqrt();
    let rope_freq = [1.0f32, 10_000.0f32];
    let mut rng = Lcg::new(0xA11C_E78Au64);
    let mut qkv = vec![0.0f32; batch * tokens * 3 * heads * head_dim];
    let mut q_gamma = vec![0.0f32; heads * head_dim];
    let mut k_gamma = vec![0.0f32; heads * head_dim];
    let mut coords = vec![0i32; tokens * 3];
    for value in &mut qkv {
        *value = rng.next_f32() * 2.0 - 1.0;
    }
    for value in &mut q_gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }
    for value in &mut k_gamma {
        *value = 0.75 + rng.next_f32() * 0.5;
    }
    for token in 0..tokens {
        coords[token * 3] = token as i32 - 5;
        coords[token * 3 + 1] = (token as i32) * 2 - 1;
        coords[token * 3 + 2] = (token as i32) * 3 + 2;
    }

    let qkv_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(qkv.clone(), [qkv.len()]),
        &device,
    )
    .reshape([batch, tokens, 3, heads, head_dim]);
    let q_gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(q_gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let k_gamma_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(k_gamma.clone(), [heads * head_dim]),
        &device,
    )
    .reshape([heads, head_dim]);
    let coords_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords.clone(), [tokens * 3]),
        &device,
    )
    .reshape([tokens, 3]);

    let (q_reference, k_reference) = multihead_qk_rms_norm_rope_from_qkv_coords_wgpu(
        qkv_t.clone(),
        q_gamma_t.clone(),
        k_gamma_t.clone(),
        coords_t.clone(),
        rope_freq,
        scale,
        eps,
    )
    .expect("fused qk rms norm rope output");
    let v_reference = qkv_t
        .clone()
        .slice([0..batch, 0..tokens, 2..3, 0..heads, 0..head_dim])
        .reshape([batch, tokens, heads, head_dim]);

    let (q_module, k_module, v_module) = multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu(
        qkv_t, q_gamma_t, k_gamma_t, coords_t, rope_freq, scale, eps,
    )
    .expect("module-layout qkv rms norm rope output");

    let q_reference = q_reference
        .permute([0, 2, 1, 3])
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let k_reference = k_reference
        .permute([0, 2, 1, 3])
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let v_reference = v_reference
        .permute([0, 2, 1, 3])
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let q_module = q_module
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let k_module = k_module
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();
    let v_module = v_module
        .reshape([rows, head_dim])
        .to_data()
        .as_slice::<f32>()
        .expect("f32")
        .to_vec();

    let mut max_abs = 0.0f32;
    for (actual, expected) in q_module.iter().zip(q_reference.iter()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    for (actual, expected) in k_module.iter().zip(k_reference.iter()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    for (actual, expected) in v_module.iter().zip(v_reference.iter()) {
        max_abs = max_abs.max((actual - expected).abs());
    }
    assert!(
        max_abs <= 1.0e-5,
        "module-layout qkv rms norm rope kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn layer_norm_affine_silu_kernel_matches_reference() {
    let _guard = env_lock_guard();
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let rows = 1024usize;
    let channels = 64usize;
    let eps = 1.0e-6f32;
    let mut rng = Lcg::new(0x4D3C2B1Au64);
    let mut input = vec![0.0f32; rows * channels];
    let mut weight = vec![0.0f32; channels];
    let mut bias = vec![0.0f32; channels];
    for value in &mut input {
        *value = rng.next_f32();
    }
    for value in &mut weight {
        *value = rng.next_f32();
    }
    for value in &mut bias {
        *value = rng.next_f32();
    }

    let input_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(input.clone(), [rows, channels]),
        &device,
    );
    let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(weight.clone(), [channels]),
        &device,
    );
    let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_data(
        TensorData::new(bias.clone(), [channels]),
        &device,
    );
    let output = layer_norm_affine_silu_forward_wgpu(input_t, weight_t, bias_t, eps)
        .expect("layer norm silu kernel output");
    let output = output.to_data().as_slice::<f32>().expect("f32").to_vec();

    let mut max_abs = 0.0f32;
    for row in 0..rows {
        let base = row * channels;
        let mut mean = 0.0f32;
        for ch in 0..channels {
            mean += input[base + ch];
        }
        mean /= channels as f32;
        let mut var = 0.0f32;
        for ch in 0..channels {
            let centered = input[base + ch] - mean;
            var += centered * centered;
        }
        var /= channels as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        for ch in 0..channels {
            let affine = (input[base + ch] - mean) * inv_std * weight[ch] + bias[ch];
            let expected = affine * (1.0 / (1.0 + (-affine).exp()));
            let actual = output[base + ch];
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    assert!(
        max_abs <= 2.0e-4,
        "layer norm affine silu kernel drift too high: max_abs={max_abs:.6e}"
    );
}

#[test]
fn wgpu_kernel_matches_cpu_flex_path() {
    let _guard = env_lock_guard();
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
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
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

    let Some(device) = wgpu_test_device() else {
        return;
    };
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
fn wgpu_single_group_specialized_kernel_matches_cpu_flex_path() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 16,
        out_channels: 24,
        kernel_d: 3,
        kernel_h: 1,
        kernel_w: 1,
        in_channels_per_group: 16,
        out_channels_per_group: 24,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(96);
    let mut rng = Lcg::new(1313);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
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

    let Some(device) = wgpu_test_device() else {
        return;
    };
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

    reset_sparse_wgpu_kernel_stats();
    let output = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t,
        neighbors_t,
        weight_t,
        bias_t,
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(1),
        },
    )
    .expect("wgpu specialized single-group kernel path");
    let output = output.to_data();
    let output = output.as_slice::<f32>().expect("f32 output");

    assert_eq!(output.len(), expected.len());
    for (idx, (lhs, rhs)) in output.iter().zip(expected.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(diff <= 1.0e-4, "mismatch at idx={idx}: lhs={lhs} rhs={rhs}");
    }

    let stats = sparse_wgpu_kernel_stats();
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.single_group_specialized_calls, 1);
}

#[test]
fn sparse_conv_hotspot_kernel_matches_reference_parity() {
    // Roadmap gate alias for specialized sparse-conv parity.
    wgpu_single_group_specialized_kernel_matches_cpu_flex_path();
}

#[test]
fn neighbor_rows_tensor_shape_is_consistent() {
    let _guard = env_lock_guard();
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
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let neighbors =
        neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device).expect("neighbors");
    let data = neighbors.to_data();
    let [rows, kernel_rows] = neighbors.dims();
    assert_eq!(rows, coords.len());
    assert_eq!(kernel_rows, 3);
    let values = data.as_slice::<i32>().expect("i32");
    assert_eq!(values.len(), rows * kernel_rows);
}

#[test]
fn neighbor_rows_cache_reuses_across_equivalent_coord_allocations() {
    let _guard = env_lock_guard();
    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();

    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 1,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(64);
    let coords_clone = coords.clone();
    let Some(device) = wgpu_test_device() else {
        return;
    };

    let first = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
        .expect("first neighbor tensor")
        .to_data();
    let second = neighbor_rows_tensor_from_coords(&cfg, coords_clone.as_slice(), &device)
        .expect("second neighbor tensor")
        .to_data();

    let first = first.as_slice::<i32>().expect("i32").to_vec();
    let second = second.as_slice::<i32>().expect("i32").to_vec();
    assert_eq!(first, second);

    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.host_builds, 0);
    assert_eq!(stats.device_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_tensor_cache_reuses_across_tensor_coord_clones() {
    let _guard = env_lock_guard();
    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();

    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 1,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(64);
    let mut coords_flat = Vec::with_capacity(coords.len() * 4);
    for coord in coords {
        coords_flat.push(coord[0] as i32);
        coords_flat.push(coord[1] as i32);
        coords_flat.push(coord[2] as i32);
        coords_flat.push(coord[3] as i32);
    }
    let Some(device) = wgpu_test_device() else {
        return;
    };
    let coords_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [64 * 4]),
        &device,
    )
    .reshape([64, 4]);

    // Tensor-path cache is keyed by device tensor identity to avoid host
    // coord materialization in canonical WGPU decode flow.
    let first = neighbor_rows_tensor_from_coords_tensor(&cfg, coords_t.clone())
        .expect("first tensor-path neighbor tensor")
        .to_data();
    let second = neighbor_rows_tensor_from_coords_tensor(&cfg, coords_t)
        .expect("second tensor-path neighbor tensor")
        .to_data();

    let first = first.as_slice::<i32>().expect("i32").to_vec();
    let second = second.as_slice::<i32>().expect("i32").to_vec();
    assert_eq!(first, second);

    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.host_builds, 0);
    assert_eq!(stats.device_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_cache_reuses_across_channel_variants_with_same_topology() {
    let _guard = env_lock_guard();
    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();

    let cfg_a = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 8,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 8,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let cfg_b = SparseSubmConvConfig {
        in_channels: 16,
        out_channels: 16,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 1,
        in_channels_per_group: 8,
        out_channels_per_group: 8,
        groups: 2,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(96);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    let first = neighbor_rows_tensor_from_coords(&cfg_a, coords.as_slice(), &device)
        .expect("first neighbor tensor")
        .to_data();
    let second = neighbor_rows_tensor_from_coords(&cfg_b, coords.as_slice(), &device)
        .expect("second neighbor tensor")
        .to_data();
    let first = first.as_slice::<i32>().expect("i32").to_vec();
    let second = second.as_slice::<i32>().expect("i32").to_vec();
    assert_eq!(first, second);

    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.cache_misses, 1);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.host_builds, 0);
    assert_eq!(stats.device_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_auto_matches_serial_hash_table_backend() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(96);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let auto_rows = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
        .expect("auto neighbor rows")
        .to_data();
    let auto_rows = auto_rows.as_slice::<i32>().expect("i32").to_vec();
    let auto_stats = neighbor_rows_build_stats();
    assert_eq!(auto_stats.cache_misses, 1);
    assert_eq!(auto_stats.device_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let serial_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::HashTableSerial,
    )
    .expect("serial hash neighbor rows")
    .to_data();
    let serial_rows = serial_rows.as_slice::<i32>().expect("i32").to_vec();
    let serial_stats = neighbor_rows_build_stats();
    // Explicit algorithm entrypoint bypasses cache accounting by design.
    assert_eq!(serial_stats.cache_misses, 0);
    assert_eq!(serial_stats.cache_hits, 0);
    assert_eq!(serial_stats.device_builds, 1);

    assert_eq!(auto_rows, serial_rows);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_device_hash_matches_scan() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(192);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let scan_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::Scan,
    )
    .expect("scan rows")
    .to_data();
    let scan_rows = scan_rows.as_slice::<i32>().expect("i32").to_vec();

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let hash_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::HashTableSerial,
    )
    .expect("hash rows")
    .to_data();
    let hash_rows = hash_rows.as_slice::<i32>().expect("i32").to_vec();

    assert_eq!(scan_rows, hash_rows);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_device_hash_matches_scan_with_duplicate_coords() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 1,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let mut coords = line_coords(128);
    coords.push([0, 17, 0, 0]);
    coords.push([0, 32, 0, 0]);
    coords.push([0, 17, 0, 0]);
    coords.push([0, 32, 0, 0]);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let scan_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::Scan,
    )
    .expect("scan rows")
    .to_data();
    let scan_rows = scan_rows.as_slice::<i32>().expect("i32").to_vec();

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let hash_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::HashTableSerial,
    )
    .expect("hash rows")
    .to_data();
    let hash_rows = hash_rows.as_slice::<i32>().expect("i32").to_vec();

    assert_eq!(scan_rows, hash_rows);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_hash_parallel_collision_stress_bounded() {
    // Roadmap gate alias for collision-heavy parity/bounded path.
    neighbor_rows_device_hash_matches_scan_with_duplicate_coords();
}

#[test]
fn neighbor_rows_hash_probe_telemetry_records_probe_stats() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(5_000);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let neighbors =
        neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device).expect("neighbors");
    assert_eq!(neighbors.dims()[0], coords.len());

    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_hash_builds, 1);
    assert_eq!(stats.device_scan_builds, 0);
    assert_eq!(stats.device_hash_rows, coords.len() as u64);
    assert_eq!(stats.device_hash_insert_fail_rows, 0);
    assert!(stats.device_hash_probe_total >= coords.len() as u64);
    assert!(stats.device_hash_probe_max >= 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_sorted_hash_matches_scan_reference() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 9,
        kernel_h: 9,
        kernel_w: 9,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(256);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let sorted_hash_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::SortedHash,
    )
    .expect("sorted hash rows")
    .to_data();
    let sorted_hash_rows = sorted_hash_rows.as_slice::<i32>().expect("i32").to_vec();
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_hash_builds, 1);

    let scan_rows = build_neighbor_rows_tensor_device_scan(&cfg, coords.as_slice(), &device)
        .expect("scan rows")
        .to_data();
    let scan_rows = scan_rows.as_slice::<i32>().expect("i32").to_vec();
    assert_eq!(scan_rows, sorted_hash_rows);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_rows_bucket_hash_matches_scan_reference() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 9,
        kernel_h: 9,
        kernel_w: 9,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(512);
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let bucket_rows = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        NeighborDeviceAlgoPreference::BucketHash,
    )
    .expect("bucket hash rows")
    .to_data();
    let bucket_rows = bucket_rows.as_slice::<i32>().expect("i32").to_vec();
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_hash_builds, 1);
    assert_eq!(stats.device_hash_insert_fail_rows, 0);

    let scan_rows = build_neighbor_rows_tensor_device_scan(&cfg, coords.as_slice(), &device)
        .expect("scan rows")
        .to_data();
    let scan_rows = scan_rows.as_slice::<i32>().expect("i32").to_vec();
    assert_eq!(scan_rows, bucket_rows);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_hash_parallel_matches_scan_parity() {
    // Roadmap gate alias for sorted-hash parallel query parity.
    neighbor_rows_sorted_hash_matches_scan_reference();
}

#[test]
fn neighbor_algo_auto_uses_kernel_aware_thresholds() {
    let _guard = env_lock_guard();
    let cfg_k3 = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let cfg_k9 = SparseSubmConvConfig {
        in_channels: 4,
        out_channels: 4,
        kernel_d: 9,
        kernel_h: 9,
        kernel_w: 9,
        in_channels_per_group: 4,
        out_channels_per_group: 4,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let Some(device) = wgpu_test_device() else {
        return;
    };

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let _ = neighbor_rows_tensor_from_coords(&cfg_k3, line_coords(2_048).as_slice(), &device)
        .expect("k3 rows=2048");
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_scan_builds, 1);
    assert_eq!(stats.device_hash_builds, 0);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let _ = neighbor_rows_tensor_from_coords(&cfg_k3, line_coords(4_096).as_slice(), &device)
        .expect("k3 rows=4096");
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_hash_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let _ = neighbor_rows_tensor_from_coords(&cfg_k9, line_coords(512).as_slice(), &device)
        .expect("k9 rows=512");
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_scan_builds, 1);
    assert_eq!(stats.device_hash_builds, 0);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
    let _ = neighbor_rows_tensor_from_coords(&cfg_k9, line_coords(1_024).as_slice(), &device)
        .expect("k9 rows=1024");
    let stats = neighbor_rows_build_stats();
    assert_eq!(stats.device_hash_builds, 1);

    clear_neighbor_rows_tensor_cache();
    reset_neighbor_rows_build_stats();
}

#[test]
fn neighbor_algo_auto_routes_bucket_hash_for_large_small_k() {
    assert_eq!(
        super::resolve_neighbor_device_algo(
            super::DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K - 1,
            27,
            NeighborDeviceAlgoPreference::Auto
        ),
        super::NeighborDeviceAlgo::SortedHash
    );
    assert_eq!(
        super::resolve_neighbor_device_algo(
            super::DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K,
            27,
            NeighborDeviceAlgoPreference::Auto
        ),
        super::NeighborDeviceAlgo::BucketHash
    );
    assert_eq!(
        super::resolve_neighbor_device_algo(
            super::DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K,
            729,
            NeighborDeviceAlgoPreference::Auto
        ),
        super::NeighborDeviceAlgo::SortedHash
    );
}

#[test]
fn neighbor_sorted_hash_search_step_resolver_uses_mid_bucket() {
    assert_eq!(
        super::resolve_neighbor_sorted_hash_search_steps(
            super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MAX
        ),
        super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL
    );
    assert_eq!(
        super::resolve_neighbor_sorted_hash_search_steps(
            super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MAX + 1
        ),
        super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM
    );
    assert_eq!(
        super::resolve_neighbor_sorted_hash_search_steps(
            super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MEDIUM_MAX
        ),
        super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM
    );
    assert_eq!(
        super::resolve_neighbor_sorted_hash_search_steps(
            super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MEDIUM_MAX + 1
        ),
        super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM
    );
    assert_eq!(
        super::resolve_neighbor_sorted_hash_search_steps(
            super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_MEDIUM_MAX + 1
        ),
        super::DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_LARGE
    );
}

#[test]
fn sparse_conv_auto_schedule_uses_splitk2_for_medium_decode_work() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 2_048, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.split_k, 2);
}

#[test]
fn sparse_conv_auto_schedule_uses_splitk4_for_larger_decode_work() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 4_096, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.split_k, 4);
}

#[test]
fn sparse_conv_auto_schedule_keeps_baseline_variant_for_common_decode_shapes() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 8_192, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.kernel_variant, SparseWgpuKernelVariant::Baseline);
}

#[test]
fn sparse_conv_auto_schedule_uses_single_group_specialized_baseline_variant() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        2_048,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup
    );
}

#[test]
fn sparse_conv_auto_schedule_uses_single_group_fused_hot_shape_variant() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 128,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        4_096,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::FusedOc4SingleGroup
    );
}

#[test]
fn sparse_conv_auto_schedule_does_not_use_single_group_specialization_for_grouped_conv() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 128,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 32,
        out_channels_per_group: 64,
        groups: 2,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        2_048,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::Baseline
    );
}

#[test]
fn sparse_conv_auto_schedule_keeps_baseline_for_rows4096_when_inner_work_is_high() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 128,
        out_channels: 256,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 128,
        out_channels_per_group: 256,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        4_096,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup
    );
}

#[test]
fn sparse_conv_auto_schedule_keeps_baseline_for_rows4096_when_oc_group_is_high() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 256,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 256,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        4_096,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup
    );
}

#[test]
fn sparse_conv_auto_schedule_caps_splitk_for_high_oc_decode_shape() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 256,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 256,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 8_192, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.split_k, 1);
}

#[test]
fn sparse_conv_auto_schedule_caps_splitk_for_mid_rows_very_high_oc_decode_shape() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 512,
        out_channels: 512,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 512,
        out_channels_per_group: 512,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 4_425, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.split_k, 1);
}

#[test]
fn sparse_conv_auto_schedule_keeps_splitk_for_small_rows_very_high_oc_shape() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 512,
        out_channels: 512,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 512,
        out_channels_per_group: 512,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let resolved =
        resolve_sparse_wgpu_forward_config(&cfg, 2_048, SparseWgpuForwardConfig::default())
            .expect("resolved forward config");
    assert_eq!(resolved.split_k, 4);
}

#[test]
fn sparse_conv_auto_schedule_keeps_baseline_for_borderline_fused_output_work_shape() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 64,
        out_channels: 256,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 64,
        out_channels_per_group: 256,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        8_192,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup
    );
    assert_eq!(resolved_internal.split_k, 1);
}

#[test]
fn sparse_conv_auto_schedule_keeps_baseline_for_high_inner_work_decode_shape() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 1024,
        out_channels: 1024,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 1024,
        out_channels_per_group: 1024,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };

    let krows = crate::kernel_rows(&cfg).expect("kernel rows");
    let resolved_internal = resolve_sparse_wgpu_forward_config_internal(
        &cfg,
        8_338,
        krows,
        SparseWgpuForwardConfig::default(),
    );
    assert_eq!(
        resolved_internal.kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup
    );
}

#[test]
fn wgpu_fused_oc4_matches_baseline_output() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 32,
        out_channels: 64,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 32,
        out_channels_per_group: 64,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(192);
    let mut rng = Lcg::new(901);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

    let Some(device) = wgpu_test_device() else {
        return;
    };
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

    let baseline = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t.clone(),
        neighbors_t.clone(),
        weight_t.clone(),
        bias_t.clone(),
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(1),
        },
    )
    .expect("baseline kernel")
    .to_data();
    let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

    let fused = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t,
        neighbors_t,
        weight_t,
        bias_t,
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::FusedOc4,
            split_k: Some(1),
        },
    )
    .expect("fused kernel")
    .to_data();
    let fused = fused.as_slice::<f32>().expect("f32");

    assert_eq!(baseline.len(), fused.len());
    for (idx, (lhs, rhs)) in fused.iter().zip(baseline.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1.0e-4,
            "fused mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
        );
    }
}

#[test]
fn wgpu_splitk_matches_default_kernel_output() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 32,
        out_channels: 64,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 32,
        out_channels_per_group: 64,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(256);
    let mut rng = Lcg::new(77);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

    let Some(device) = wgpu_test_device() else {
        return;
    };
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

    let baseline = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t.clone(),
        neighbors_t.clone(),
        weight_t.clone(),
        bias_t.clone(),
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(1),
        },
    )
    .expect("baseline kernel")
    .to_data();
    let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

    let splitk = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t,
        neighbors_t,
        weight_t,
        bias_t,
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(4),
        },
    )
    .expect("splitk kernel")
    .to_data();
    let splitk = splitk.as_slice::<f32>().expect("f32");

    assert_eq!(baseline.len(), splitk.len());
    for (idx, (lhs, rhs)) in splitk.iter().zip(baseline.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1.0e-4,
            "split-k mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
        );
    }
}

#[test]
fn wgpu_im2col_matmul_matches_baseline_output() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 16,
        out_channels: 24,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 16,
        out_channels_per_group: 24,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(96);
    let mut rng = Lcg::new(9187);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

    let Some(device) = wgpu_test_device() else {
        return;
    };
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

    let baseline = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t.clone(),
        neighbors_t.clone(),
        weight_t.clone(),
        bias_t.clone(),
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(1),
        },
    )
    .expect("baseline kernel")
    .to_data();
    let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

    let im2col = sparse_subm_conv_forward_wgpu_im2col_matmul(
        &cfg,
        input_t.clone(),
        neighbors_t.clone(),
        weight_t.clone(),
        bias_t.clone(),
    )
    .expect("im2col matmul kernel")
    .to_data();
    let im2col = im2col.as_slice::<f32>().expect("f32").to_vec();

    assert_eq!(baseline.len(), im2col.len());
    for (idx, (lhs, rhs)) in im2col.iter().zip(baseline.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1.0e-3,
            "im2col mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
        );
    }

    let im2col_f16 = sparse_subm_conv_forward_wgpu_im2col_matmul_fast_f16(
        &cfg,
        input_t,
        neighbors_t,
        weight_t,
        bias_t,
    )
    .expect("im2col f16 matmul kernel")
    .to_data();
    let im2col_f16 = im2col_f16.as_slice::<f32>().expect("f32");
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (actual, expected) in im2col_f16.iter().zip(im2col.iter()) {
        let diff = (actual - expected).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    let mean_abs = sum_abs / im2col_f16.len().max(1) as f32;
    assert!(
        mean_abs <= 3.0e-2 && max_abs <= 2.5e-1,
        "im2col f16 drift too high: mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}"
    );
}

#[test]
fn wgpu_fused_splitk_matches_baseline_output() {
    let _guard = env_lock_guard();
    let cfg = SparseSubmConvConfig {
        in_channels: 32,
        out_channels: 64,
        kernel_d: 3,
        kernel_h: 3,
        kernel_w: 3,
        in_channels_per_group: 32,
        out_channels_per_group: 64,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    };
    let coords = line_coords(256);
    let mut rng = Lcg::new(1457);
    let input: Vec<f32> = (0..coords.len() * cfg.in_channels)
        .map(|_| rng.next_f32())
        .collect();
    let weight_len =
        cfg.out_channels * cfg.kernel_d * cfg.kernel_h * cfg.kernel_w * cfg.in_channels_per_group;
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

    let Some(device) = wgpu_test_device() else {
        return;
    };
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

    let baseline = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t.clone(),
        neighbors_t.clone(),
        weight_t.clone(),
        bias_t.clone(),
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::Baseline,
            split_k: Some(1),
        },
    )
    .expect("baseline kernel")
    .to_data();
    let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

    let fused_split = sparse_subm_conv_forward_wgpu_with_config(
        &cfg,
        input_t,
        neighbors_t,
        weight_t,
        bias_t,
        SparseWgpuForwardConfig {
            kernel_variant: SparseWgpuKernelVariant::FusedOc4,
            split_k: Some(4),
        },
    )
    .expect("fused split-k kernel")
    .to_data();
    let fused_split = fused_split.as_slice::<f32>().expect("f32");

    assert_eq!(baseline.len(), fused_split.len());
    for (idx, (lhs, rhs)) in fused_split.iter().zip(baseline.iter()).enumerate() {
        let diff = (lhs - rhs).abs();
        assert!(
            diff <= 1.0e-4,
            "fused split-k mismatch at idx={idx}: lhs={lhs} rhs={rhs} diff={diff}"
        );
    }
}

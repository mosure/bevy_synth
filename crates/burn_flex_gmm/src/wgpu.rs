use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use burn::tensor::{DType, Int, Shape, Tensor as BurnTensor, TensorData, TensorPrimitive};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};

use crate::{SparseSubmConvConfig, build_neighbor_rows, kernel_rows};

/// Default WGPU backend type used by the tensor convenience wrappers.
pub type DefaultWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

const DEFAULT_NEIGHBOR_CACHE_MAX: usize = 128;
const INVALID_NEIGHBOR: i32 = -1;
const HASH_SLOT_EMPTY: i32 = -1;
const DEFAULT_NEIGHBOR_HASH_LOAD_FACTOR: usize = 2;
const FUSED_OC_TILE: u32 = 4;

static NEIGHBOR_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_BUILDS_HOST: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_BUILDS_DEVICE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NeighborRowsBuildStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub host_builds: u64,
    pub device_builds: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum NeighborBuildBackend {
    Host,
    Device,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NeighborDeviceAlgo {
    Scan,
    Hash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NeighborHashBuildMode {
    Auto,
    Host,
    Wgsl,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseConvKernelVariant {
    Baseline,
    FusedOc4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseWgpuKernelVariant {
    Auto,
    Baseline,
    FusedOc4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseWgpuForwardConfig {
    pub kernel_variant: SparseWgpuKernelVariant,
    pub split_k: Option<usize>,
}

impl Default for SparseWgpuForwardConfig {
    fn default() -> Self {
        Self {
            kernel_variant: SparseWgpuKernelVariant::Auto,
            split_k: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct NeighborConfigCacheKey {
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    axis_order: [usize; 3],
    axis_sign: [i32; 3],
}

impl From<&SparseSubmConvConfig> for NeighborConfigCacheKey {
    fn from(config: &SparseSubmConvConfig) -> Self {
        Self {
            kernel_d: config.kernel_d,
            kernel_h: config.kernel_h,
            kernel_w: config.kernel_w,
            axis_order: config.axis_order,
            axis_sign: config.axis_sign,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct NeighborRowsCacheKey {
    config: NeighborConfigCacheKey,
    backend: NeighborBuildBackend,
    rows: usize,
    coords_hash: u64,
    device_key: String,
}

thread_local! {
    static NEIGHBOR_TENSOR_CACHE: RefCell<HashMap<NeighborRowsCacheKey, BurnTensor<DefaultWgpuBackend, 2, Int>>> =
        RefCell::new(HashMap::new());
}

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

#[cube(launch_unchecked)]
fn sparse_subm_conv_fused_oc4_kernel(
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

    let mut acc_0 = Line::new(0.0);
    let mut acc_1 = Line::new(0.0);
    let mut acc_2 = Line::new(0.0);
    let mut acc_3 = Line::new(0.0);
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
        let safe_neighbor = Max::max(neighbor, Line::new(0));
        let in_row = u32::cast_from(safe_neighbor);
        let invalid = neighbor.equal(Line::new(-1));
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
                acc_0 += select_many(invalid, Line::new(0.0), term_0);
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
                acc_1 += select_many(invalid, Line::new(0.0), term_1);
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
                acc_2 += select_many(invalid, Line::new(0.0), term_2);
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
                acc_3 += select_many(invalid, Line::new(0.0), term_3);
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

#[cube(launch_unchecked)]
fn sparse_subm_conv_splitk_partial_kernel(
    input: &Tensor<Line<f32>>,
    neighbor_rows: &Tensor<Line<i32>>,
    weight: &Tensor<Line<f32>>,
    partial: &mut Tensor<Line<f32>>,
    out_channels: &u32,
    kernel_rows: &u32,
    in_channels: &u32,
    in_channels_per_group: &u32,
    out_channels_per_group: &u32,
    output_elements: &u32,
    split_k: &u32,
) {
    if ABSOLUTE_POS >= partial.len() {
        terminate!();
    }

    let partial_idx = ABSOLUTE_POS;
    let split_idx = partial_idx / *output_elements;
    let out_idx = partial_idx % *output_elements;
    if split_idx >= *split_k {
        partial[partial_idx] = Line::new(0.0);
        terminate!();
    }

    let row = out_idx / *out_channels;
    let out_channel = out_idx % *out_channels;
    let group = out_channel / *out_channels_per_group;
    let in_group_base = group * *in_channels_per_group;
    let chunk = (*kernel_rows).div_ceil(*split_k);
    let kernel_start = split_idx * chunk;
    let kernel_end = Min::min(kernel_start + chunk, *kernel_rows);

    let mut acc = Line::new(0.0);
    for kernel_idx in kernel_start..kernel_end {
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

    partial[partial_idx] = acc;
}

#[cube(launch_unchecked)]
fn sparse_subm_conv_splitk_partial_fused_oc4_kernel(
    input: &Tensor<Line<f32>>,
    neighbor_rows: &Tensor<Line<i32>>,
    weight: &Tensor<Line<f32>>,
    partial: &mut Tensor<Line<f32>>,
    out_channels: &u32,
    kernel_rows: &u32,
    in_channels: &u32,
    in_channels_per_group: &u32,
    out_channels_per_group: &u32,
    output_elements: &u32,
    split_k: &u32,
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
    let kernel_end = Min::min(kernel_start + chunk, *kernel_rows);

    let mut acc_0 = Line::new(0.0);
    let mut acc_1 = Line::new(0.0);
    let mut acc_2 = Line::new(0.0);
    let mut acc_3 = Line::new(0.0);

    for kernel_idx in kernel_start..kernel_end {
        let neighbor = neighbor_rows[row * *kernel_rows + kernel_idx];
        let safe_neighbor = Max::max(neighbor, Line::new(0));
        let in_row = u32::cast_from(safe_neighbor);
        let invalid = neighbor.equal(Line::new(-1));

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
                acc_0 += select_many(invalid, Line::new(0.0), term_0);
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
                acc_1 += select_many(invalid, Line::new(0.0), term_1);
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
                acc_2 += select_many(invalid, Line::new(0.0), term_2);
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
                acc_3 += select_many(invalid, Line::new(0.0), term_3);
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
fn sparse_subm_conv_splitk_finalize_kernel(
    partial: &Tensor<Line<f32>>,
    bias: &Tensor<Line<f32>>,
    output: &mut Tensor<Line<f32>>,
    out_channels: &u32,
    output_elements: &u32,
    split_k: &u32,
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

#[cube(launch_unchecked)]
fn neighbor_rows_from_coords_kernel(
    coords: &Tensor<Line<i32>>,
    offsets: &Tensor<Line<i32>>,
    neighbor_rows: &mut Tensor<Line<i32>>,
    rows: &u32,
    kernel_rows: &u32,
) {
    if ABSOLUTE_POS >= neighbor_rows.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_row = out_idx / *kernel_rows;
    let kernel_idx = out_idx % *kernel_rows;
    let coord_base = out_row * 4;
    let batch = coords[coord_base];
    let ox = coords[coord_base + 1];
    let oy = coords[coord_base + 2];
    let oz = coords[coord_base + 3];

    let offset_base = kernel_idx * 3;
    let nx = ox + offsets[offset_base];
    let ny = oy + offsets[offset_base + 1];
    let nz = oz + offsets[offset_base + 2];

    let mut found = Line::new(INVALID_NEIGHBOR);
    for in_row in 0..*rows {
        let src = in_row * 4;
        let same_batch = coords[src].equal(batch);
        let same_x = coords[src + 1].equal(nx);
        let same_y = coords[src + 2].equal(ny);
        let same_z = coords[src + 3].equal(nz);
        let same_xy = select_many(same_batch, same_x, Line::new(false));
        let same_xyz = select_many(same_xy, same_y, Line::new(false));
        let same = select_many(same_xyz, same_z, Line::new(false));
        let should_set = select_many(
            same,
            found.equal(Line::new(INVALID_NEIGHBOR)),
            Line::new(false),
        );
        let in_row_i32 = Line::new(i32::cast_from(in_row));
        found = select_many(should_set, in_row_i32, found);
    }

    found = select_many(
        nx.less_than(Line::new(0)),
        Line::new(INVALID_NEIGHBOR),
        found,
    );
    found = select_many(
        ny.less_than(Line::new(0)),
        Line::new(INVALID_NEIGHBOR),
        found,
    );
    found = select_many(
        nz.less_than(Line::new(0)),
        Line::new(INVALID_NEIGHBOR),
        found,
    );
    neighbor_rows[out_idx] = found;
}

#[cube]
fn spatial_hash_u32(batch: i32, x: i32, y: i32, z: i32) -> u32 {
    let b = u32::cast_from(batch);
    let xx = u32::cast_from(x);
    let yy = u32::cast_from(y);
    let zz = u32::cast_from(z);
    let mut hash = b * 0x9e37_79b1u32;
    hash ^= xx * 0x85eb_ca77u32;
    hash ^= yy * 0xc2b2_ae3du32;
    hash ^= zz * 0x27d4_eb2fu32;
    hash
}

#[cube(launch_unchecked)]
fn neighbor_hash_reset_kernel(table_rows: &mut Array<i32>, fill: &i32) {
    if ABSOLUTE_POS >= table_rows.len() {
        terminate!();
    }
    table_rows[ABSOLUTE_POS] = *fill;
}

#[cube(launch_unchecked)]
fn neighbor_hash_build_serial_kernel(
    coords: &Array<i32>,
    table_rows: &mut Array<i32>,
    table_coords: &mut Array<i32>,
    overflow_flag: &mut Array<i32>,
    rows: &u32,
    table_mask: &u32,
    max_probe: &u32,
) {
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    for row in 0..*rows {
        let coord_base = row * 4;
        let batch = coords[coord_base];
        let x = coords[coord_base + 1];
        let y = coords[coord_base + 2];
        let z = coords[coord_base + 3];
        let hash = spatial_hash_u32(batch, x, y, z);
        let row_i32 = i32::cast_from(row);
        let inserted = RuntimeCell::<i32>::new(0);

        for probe in 0..*max_probe {
            if inserted.read() == 0 {
                let slot = (hash + probe) & *table_mask;
                let slot_row = table_rows[slot];
                if slot_row == HASH_SLOT_EMPTY {
                    table_rows[slot] = row_i32;
                    let dst = slot * 4;
                    table_coords[dst] = batch;
                    table_coords[dst + 1] = x;
                    table_coords[dst + 2] = y;
                    table_coords[dst + 3] = z;
                    inserted.store(1);
                } else {
                    let dst = slot * 4;
                    let same = table_coords[dst] == batch
                        && table_coords[dst + 1] == x
                        && table_coords[dst + 2] == y
                        && table_coords[dst + 3] == z;
                    if same {
                        if row_i32 < slot_row {
                            table_rows[slot] = row_i32;
                        }
                        inserted.store(1);
                    }
                }
            }
        }
        if inserted.read() == 0 {
            overflow_flag[0] = 1;
        }
    }
}

#[cube(launch_unchecked)]
fn neighbor_hash_query_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    table_rows: &Array<i32>,
    table_coords: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    kernel_rows: &u32,
    table_mask: &u32,
    max_probe: &u32,
) {
    if ABSOLUTE_POS >= neighbor_rows.len() {
        terminate!();
    }

    let out_idx = ABSOLUTE_POS;
    let out_row = out_idx / *kernel_rows;
    let kernel_idx = out_idx % *kernel_rows;
    let coord_base = out_row * 4;
    let batch = coords[coord_base];
    let ox = coords[coord_base + 1];
    let oy = coords[coord_base + 2];
    let oz = coords[coord_base + 3];

    let offset_base = kernel_idx * 3;
    let nx = ox + offsets[offset_base];
    let ny = oy + offsets[offset_base + 1];
    let nz = oz + offsets[offset_base + 2];
    if nx < 0 || ny < 0 || nz < 0 {
        neighbor_rows[out_idx] = INVALID_NEIGHBOR;
        terminate!();
    }

    let hash = spatial_hash_u32(batch, nx, ny, nz);
    let found = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);
    let active = RuntimeCell::<i32>::new(1);
    for probe in 0..*max_probe {
        if active.read() == 1 {
            let slot = (hash + probe) & *table_mask;
            let state = table_rows[slot];
            if state == HASH_SLOT_EMPTY {
                active.store(0);
            } else {
                let table_base = slot * 4;
                if table_coords[table_base] == batch
                    && table_coords[table_base + 1] == nx
                    && table_coords[table_base + 2] == ny
                    && table_coords[table_base + 3] == nz
                {
                    found.store(state);
                    active.store(0);
                }
            }
        }
    }

    neighbor_rows[out_idx] = found.read();
}

fn resolve_cube_dim() -> CubeDim {
    let default = CubeDim::default();
    let requested = std::env::var("BURN_FLEX_GMM_WGPU_WORKGROUP_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .filter(|value| (32..=1024).contains(value))
        .unwrap_or(default.num_elems());
    CubeDim::new_1d(requested)
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
    } else if let Ok(raw) = std::env::var("BURN_FLEX_GMM_WGPU_SPLIT_K") {
        let value = raw.trim().to_ascii_lowercase();
        if value == "off" || value == "0" || value == "1" {
            1
        } else if value != "auto" {
            value
                .parse::<usize>()
                .ok()
                .map(|parsed| parsed.clamp(1, max_split))
                .unwrap_or(1)
        } else {
            let k_in = kernel_rows.saturating_mul(config.in_channels_per_group);
            let work = rows
                .saturating_mul(config.out_channels_per_group)
                .saturating_mul(k_in);
            if work >= 64 * 1024 * 1024 {
                4
            } else if work >= 24 * 1024 * 1024 {
                2
            } else {
                1
            }
        }
    } else {
        let k_in = kernel_rows.saturating_mul(config.in_channels_per_group);
        let work = rows
            .saturating_mul(config.out_channels_per_group)
            .saturating_mul(k_in);
        if work >= 64 * 1024 * 1024 {
            4
        } else if work >= 24 * 1024 * 1024 {
            2
        } else {
            1
        }
    };

    let output_elements = rows.saturating_mul(config.out_channels);
    let output_bytes = output_elements.saturating_mul(core::mem::size_of::<f32>());
    let max_partial_bytes = std::env::var("BURN_FLEX_GMM_WGPU_SPLIT_K_MAX_PARTIAL_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256 * 1024 * 1024);
    while split > 1 {
        let partial_bytes = output_bytes.saturating_mul(split);
        if partial_bytes <= max_partial_bytes {
            break;
        }
        split -= 1;
    }
    split.max(1)
}

fn resolve_sparse_conv_kernel_variant(
    config: &SparseSubmConvConfig,
    rows: usize,
    kernel_rows: usize,
    kernel_override: SparseWgpuKernelVariant,
) -> SparseConvKernelVariant {
    match kernel_override {
        SparseWgpuKernelVariant::Baseline => return SparseConvKernelVariant::Baseline,
        SparseWgpuKernelVariant::FusedOc4 => return SparseConvKernelVariant::FusedOc4,
        SparseWgpuKernelVariant::Auto => {}
    }

    if let Ok(raw) = std::env::var("BURN_FLEX_GMM_WGPU_KERNEL") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "baseline" | "default" | "legacy" => return SparseConvKernelVariant::Baseline,
            "fused" | "fused_oc4" => return SparseConvKernelVariant::FusedOc4,
            "auto" => {}
            _ => {}
        }
    }

    let inner_work = kernel_rows.saturating_mul(config.in_channels_per_group);
    let output_work = rows.saturating_mul(config.out_channels_per_group);
    if config.out_channels_per_group >= FUSED_OC_TILE as usize
        && config.out_channels >= FUSED_OC_TILE as usize
        && inner_work >= 64
        && output_work >= 2048
    {
        SparseConvKernelVariant::FusedOc4
    } else {
        SparseConvKernelVariant::Baseline
    }
}

fn resolve_neighbor_backend(rows: usize, kernel_rows: usize) -> NeighborBuildBackend {
    if let Ok(raw) = std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "host" | "cpu" => return NeighborBuildBackend::Host,
            "device" | "gpu" => return NeighborBuildBackend::Device,
            _ => {}
        }
    }

    let work = rows.saturating_mul(kernel_rows);
    match resolve_neighbor_hash_build_mode() {
        NeighborHashBuildMode::Host => {
            if work > 131_072 {
                NeighborBuildBackend::Host
            } else {
                NeighborBuildBackend::Device
            }
        }
        NeighborHashBuildMode::Wgsl => NeighborBuildBackend::Device,
        NeighborHashBuildMode::Auto => {
            // For large workloads, keep the robust host build path by default.
            // WGSL hash build remains available via explicit env override.
            if work > 131_072 {
                NeighborBuildBackend::Host
            } else {
                NeighborBuildBackend::Device
            }
        }
    }
}

fn resolve_neighbor_device_algo(rows: usize, kernel_rows: usize) -> NeighborDeviceAlgo {
    if let Ok(raw) = std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "scan" => return NeighborDeviceAlgo::Scan,
            "hash" => return NeighborDeviceAlgo::Hash,
            "auto" => {}
            _ => {}
        }
    }

    let work = rows.saturating_mul(kernel_rows);
    if work >= 131_072 {
        NeighborDeviceAlgo::Hash
    } else {
        NeighborDeviceAlgo::Scan
    }
}

fn resolve_neighbor_hash_build_mode() -> NeighborHashBuildMode {
    if let Ok(raw) = std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_BUILD") {
        match raw.trim().to_ascii_lowercase().as_str() {
            "host" | "cpu" => return NeighborHashBuildMode::Host,
            "wgsl" | "device" | "gpu" => return NeighborHashBuildMode::Wgsl,
            "auto" => return NeighborHashBuildMode::Auto,
            _ => {}
        }
    }
    NeighborHashBuildMode::Auto
}

fn resolve_neighbor_hash_load_factor() -> usize {
    std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_LOAD_FACTOR")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value >= 2)
        .unwrap_or(DEFAULT_NEIGHBOR_HASH_LOAD_FACTOR)
}

fn resolve_neighbor_hash_table_size(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let load_factor = resolve_neighbor_hash_load_factor();
    let min_capacity = rows.saturating_mul(load_factor);
    let capacity = min_capacity.next_power_of_two();
    capacity.max(64)
}

fn resolve_neighbor_hash_max_probe(table_size: usize) -> usize {
    // Large default probe windows can cause pathological WGSL loop cost and buffer invalidation.
    // Keep a conservative cap by default; callers can raise/lower with env override.
    let limit = std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_HASH_MAX_PROBE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    limit.min(table_size).max(1)
}

fn neighbor_cache_max_entries() -> usize {
    std::env::var("BURN_FLEX_GMM_WGPU_NEIGHBOR_CACHE_MAX")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_NEIGHBOR_CACHE_MAX)
}

fn trim_cache(cache: &mut HashMap<NeighborRowsCacheKey, BurnTensor<DefaultWgpuBackend, 2, Int>>) {
    let max = neighbor_cache_max_entries();
    while cache.len() > max {
        let Some(key) = cache.keys().next().cloned() else {
            break;
        };
        cache.remove(&key);
    }
}

fn hash_coords(coords: &[[u32; 4]]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for coord in coords {
        for value in coord {
            hash ^= *value as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash ^= coords.len() as u64;
    hash.wrapping_mul(0x0000_0100_0000_01b3)
}

fn kernel_offsets(config: &SparseSubmConvConfig) -> Vec<[i32; 3]> {
    let center_d = (config.kernel_d / 2) as i32;
    let center_h = (config.kernel_h / 2) as i32;
    let center_w = (config.kernel_w / 2) as i32;
    let mut offsets = Vec::with_capacity(
        config
            .kernel_d
            .saturating_mul(config.kernel_h)
            .saturating_mul(config.kernel_w),
    );
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
    offsets
}

fn neighbor_cache_key(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    backend: NeighborBuildBackend,
) -> NeighborRowsCacheKey {
    NeighborRowsCacheKey {
        config: NeighborConfigCacheKey::from(config),
        backend,
        rows: coords.len(),
        coords_hash: hash_coords(coords),
        device_key: format!("{device:?}"),
    }
}

fn build_neighbor_rows_tensor_host(
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

    Ok(BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(neighbor_rows, [rows * kernel_rows]),
        device,
    )
    .reshape([rows, kernel_rows]))
}

fn flatten_coords_i32(coords: &[[u32; 4]]) -> Result<Vec<i32>, String> {
    let mut coords_flat = Vec::with_capacity(coords.len() * 4);
    for coord in coords.iter().copied() {
        for value in coord {
            let converted = i32::try_from(value).map_err(|_| {
                format!("coord value {value} exceeds i32::MAX for device neighbor kernel")
            })?;
            coords_flat.push(converted);
        }
    }
    Ok(coords_flat)
}

fn spatial_hash_host_u32(batch: i32, x: i32, y: i32, z: i32) -> u32 {
    let b = batch as u32;
    let xx = x as u32;
    let yy = y as u32;
    let zz = z as u32;
    let mut hash = b.wrapping_mul(0x9e37_79b1u32);
    hash ^= xx.wrapping_mul(0x85eb_ca77u32);
    hash ^= yy.wrapping_mul(0xc2b2_ae3du32);
    hash ^= zz.wrapping_mul(0x27d4_eb2fu32);
    hash
}

fn build_coord_hash_table_host(
    coords_flat: &[i32],
    rows: usize,
    table_size: usize,
) -> Result<(Vec<i32>, Vec<i32>), String> {
    let mut table_rows = vec![HASH_SLOT_EMPTY; table_size];
    let mut table_coords = vec![0i32; table_size * 4];
    let mask = table_size - 1;
    for row in 0..rows {
        let row_base = row * 4;
        let batch = coords_flat[row_base];
        let x = coords_flat[row_base + 1];
        let y = coords_flat[row_base + 2];
        let z = coords_flat[row_base + 3];
        let hash = spatial_hash_host_u32(batch, x, y, z) as usize;
        let row_i32 = i32::try_from(row)
            .map_err(|_| "neighbor row index exceeds i32::MAX in hash table build".to_string())?;
        let mut inserted = false;
        for probe in 0..table_size {
            let slot = (hash + probe) & mask;
            let slot_row = table_rows[slot];
            if slot_row == HASH_SLOT_EMPTY {
                table_rows[slot] = row_i32;
                let dst = slot * 4;
                table_coords[dst] = batch;
                table_coords[dst + 1] = x;
                table_coords[dst + 2] = y;
                table_coords[dst + 3] = z;
                inserted = true;
                break;
            }
            let dst = slot * 4;
            let same = table_coords[dst] == batch
                && table_coords[dst + 1] == x
                && table_coords[dst + 2] == y
                && table_coords[dst + 3] == z;
            if same {
                if row_i32 < slot_row {
                    table_rows[slot] = row_i32;
                }
                inserted = true;
                break;
            }
        }
        if !inserted {
            return Err(format!(
                "neighbor hash table insertion failed at row {row} with table_size={table_size}"
            ));
        }
    }
    Ok((table_rows, table_coords))
}

fn build_neighbor_rows_tensor_device_scan(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    let coords_flat = flatten_coords_i32(coords)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    )
    .reshape([rows, 4]);
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        device,
    )
    .reshape([kernel_rows, 3]);

    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;

    let coords_p = coords_t.into_primitive();
    let offsets_p = offsets_t.into_primitive();
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([rows, kernel_rows]),
        coords_p.client.empty(output_bytes),
        DType::I32,
    );

    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(output_elements, cube_dim);
    unsafe {
        neighbor_rows_from_coords_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            cube_count,
            cube_dim,
            coords_p.as_tensor_arg::<i32>(1),
            offsets_p.as_tensor_arg::<i32>(1),
            output.as_tensor_arg::<i32>(1),
            ScalarArg::new(rows as u32),
            ScalarArg::new(kernel_rows as u32),
        );
    }

    Ok(BurnTensor::from_primitive(output))
}

fn build_neighbor_rows_tensor_device_hash_host_table(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    let coords_flat = flatten_coords_i32(coords)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let table_size = resolve_neighbor_hash_table_size(rows);
    if table_size > i32::MAX as usize {
        return Err("neighbor hash table size exceeds i32::MAX entries".to_string());
    }
    let table_coords_elements = table_size
        .checked_mul(4)
        .ok_or_else(|| "neighbor hash coordinate table size overflow".to_string())?;
    let (table_rows_host, table_coords_host) =
        build_coord_hash_table_host(coords_flat.as_slice(), rows, table_size)?;
    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    );
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        device,
    );
    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;

    let coords_p = coords_t.into_primitive();
    let offsets_p = offsets_t.into_primitive();
    let table_rows_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(table_rows_host, [table_size]),
        device,
    );
    let table_coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(table_coords_host, [table_coords_elements]),
        device,
    );
    let table_rows_p = table_rows_t.into_primitive();
    let table_coords_p = table_coords_t.into_primitive();
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );

    let table_mask = u32::try_from(table_size - 1)
        .map_err(|_| "neighbor hash table mask exceeds u32::MAX".to_string())?;
    let max_probe = u32::try_from(resolve_neighbor_hash_max_probe(table_size))
        .map_err(|_| "neighbor hash max probe exceeds u32::MAX".to_string())?;
    let cube_dim = resolve_cube_dim();
    let query_count = calculate_cube_count_elemwise(output_elements, cube_dim);
    unsafe {
        neighbor_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.as_array_arg::<i32>(1),
            offsets_p.as_array_arg::<i32>(1),
            table_rows_p.as_array_arg::<i32>(1),
            table_coords_p.as_array_arg::<i32>(1),
            output.as_array_arg::<i32>(1),
            ScalarArg::new(kernel_rows as u32),
            ScalarArg::new(table_mask),
            ScalarArg::new(max_probe),
        );
    }

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device_hash_wgsl_table(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    let coords_flat = flatten_coords_i32(coords)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let table_size = resolve_neighbor_hash_table_size(rows);
    if table_size > i32::MAX as usize {
        return Err("neighbor hash table size exceeds i32::MAX entries".to_string());
    }
    let table_coords_elements = table_size
        .checked_mul(4)
        .ok_or_else(|| "neighbor hash coordinate table size overflow".to_string())?;
    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let table_rows_bytes = table_size
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash row table byte size overflow".to_string())?;
    let table_coords_bytes = table_coords_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash coord table byte size overflow".to_string())?;

    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    );
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        device,
    );
    let coords_p = coords_t.into_primitive();
    let offsets_p = offsets_t.into_primitive();
    let table_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([table_size]),
        coords_p.client.empty(table_rows_bytes),
        DType::I32,
    );
    let table_coords = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([table_coords_elements]),
        coords_p.client.empty(table_coords_bytes),
        DType::I32,
    );
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );
    let overflow_flag = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(vec![0i32], [1]),
        device,
    );
    let overflow_p = overflow_flag.into_primitive();

    let table_mask = u32::try_from(table_size - 1)
        .map_err(|_| "neighbor hash table mask exceeds u32::MAX".to_string())?;
    let max_probe = u32::try_from(resolve_neighbor_hash_max_probe(table_size))
        .map_err(|_| "neighbor hash max probe exceeds u32::MAX".to_string())?;
    let rows_u32 =
        u32::try_from(rows).map_err(|_| "neighbor row count exceeds u32::MAX".to_string())?;

    let cube_dim = resolve_cube_dim();
    let reset_count = calculate_cube_count_elemwise(table_size, cube_dim);
    unsafe {
        neighbor_hash_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_count,
            cube_dim,
            table_rows.as_array_arg::<i32>(1),
            ScalarArg::new(HASH_SLOT_EMPTY),
        );
    }

    let build_count = calculate_cube_count_elemwise(1, cube_dim);
    unsafe {
        // TODO(#2 wgsl-hash-build): replace serial device hash insertion with a
        // parallel subgroup-friendly hash build once adapter-safe atomics are available.
        neighbor_hash_build_serial_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            build_count,
            cube_dim,
            coords_p.as_array_arg::<i32>(1),
            table_rows.as_array_arg::<i32>(1),
            table_coords.as_array_arg::<i32>(1),
            overflow_p.as_array_arg::<i32>(1),
            ScalarArg::new(rows_u32),
            ScalarArg::new(table_mask),
            ScalarArg::new(max_probe),
        );
    }
    let overflow_value = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_primitive(overflow_p)
        .into_data()
        .convert::<i32>()
        .to_vec::<i32>()
        .map_err(|err| format!("failed to read neighbor hash overflow flag: {err:?}"))?
        .into_iter()
        .next()
        .unwrap_or(1);
    if overflow_value != 0 {
        return Err(format!(
            "neighbor hash serial build exceeded max_probe={max_probe} for rows={rows} table_size={table_size}"
        ));
    }

    let query_count = calculate_cube_count_elemwise(output_elements, cube_dim);
    unsafe {
        neighbor_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.as_array_arg::<i32>(1),
            offsets_p.as_array_arg::<i32>(1),
            table_rows.as_array_arg::<i32>(1),
            table_coords.as_array_arg::<i32>(1),
            output.as_array_arg::<i32>(1),
            ScalarArg::new(kernel_rows as u32),
            ScalarArg::new(table_mask),
            ScalarArg::new(max_probe),
        );
    }

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device_hash(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let table_size = resolve_neighbor_hash_table_size(rows);
    match resolve_neighbor_hash_build_mode() {
        NeighborHashBuildMode::Host => {
            build_neighbor_rows_tensor_device_hash_host_table(config, coords, device)
        }
        NeighborHashBuildMode::Wgsl => {
            build_neighbor_rows_tensor_device_hash_wgsl_table(config, coords, device)
        }
        NeighborHashBuildMode::Auto => {
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                build_neighbor_rows_tensor_device_hash_wgsl_table(config, coords, device)
            }));
            match attempt {
                Ok(Ok(tensor)) => Ok(tensor),
                Ok(Err(err)) => {
                    eprintln!(
                        "burn_flex_gmm: device hash-table build failed for rows={rows}, table_size={table_size}; falling back to host table build: {err}"
                    );
                    build_neighbor_rows_tensor_device_hash_host_table(config, coords, device)
                }
                Err(_) => {
                    eprintln!(
                        "burn_flex_gmm: device hash-table build panicked for rows={rows}, table_size={table_size}; falling back to host table build"
                    );
                    build_neighbor_rows_tensor_device_hash_host_table(config, coords, device)
                }
            }
        }
    }
}

fn build_neighbor_rows_tensor_device(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let kernel_rows = kernel_rows(config)?;
    if rows == 0 || kernel_rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2, Int>::zeros(
            [rows, kernel_rows],
            device,
        ));
    }
    if rows > i32::MAX as usize {
        return Err("sparse conv row count exceeds i32::MAX for neighbor kernel".to_string());
    }

    match resolve_neighbor_device_algo(rows, kernel_rows) {
        NeighborDeviceAlgo::Scan => build_neighbor_rows_tensor_device_scan(config, coords, device),
        // TODO(#2 wgsl-hash-build): make `Hash` dispatch to fully device-resident hash build+query
        // when the experimental WGSL path is enabled; fallback to current mixed host/device path.
        NeighborDeviceAlgo::Hash => build_neighbor_rows_tensor_device_hash(config, coords, device),
    }
}

pub fn clear_neighbor_rows_tensor_cache() {
    NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow_mut().clear());
}

pub fn reset_neighbor_rows_build_stats() {
    NEIGHBOR_CACHE_HITS.store(0, Ordering::Relaxed);
    NEIGHBOR_CACHE_MISSES.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_HOST.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_DEVICE.store(0, Ordering::Relaxed);
}

pub fn neighbor_rows_build_stats() -> NeighborRowsBuildStats {
    NeighborRowsBuildStats {
        cache_hits: NEIGHBOR_CACHE_HITS.load(Ordering::Relaxed),
        cache_misses: NEIGHBOR_CACHE_MISSES.load(Ordering::Relaxed),
        host_builds: NEIGHBOR_BUILDS_HOST.load(Ordering::Relaxed),
        device_builds: NEIGHBOR_BUILDS_DEVICE.load(Ordering::Relaxed),
    }
}

/// Launch sparse submanifold convolution directly on CubeCL tensors.
///
/// All tensors stay device-resident during execution.
fn sparse_subm_conv_forward_cubecl_impl<R: CubeRuntime>(
    config: &SparseSubmConvConfig,
    input: CubeTensor<R>,
    neighbor_rows: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
    forward: SparseWgpuForwardConfig,
) -> Result<CubeTensor<R>, String> {
    validate_tensor_shapes(config, &input, &neighbor_rows, &weight, &bias)?;

    let query_rows = neighbor_rows.shape.dims[0];
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
    let split_k = resolve_split_k(config, query_rows, kernel_rows, forward.split_k);
    let kernel_variant =
        resolve_sparse_conv_kernel_variant(config, query_rows, kernel_rows, forward.kernel_variant);
    let cube_dim = resolve_cube_dim();
    if split_k <= 1 {
        match kernel_variant {
            SparseConvKernelVariant::Baseline => {
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
            }
            SparseConvKernelVariant::FusedOc4 => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE as usize) as u32;
                let output_blocks = query_rows
                    .checked_mul(blocks_per_row as usize)
                    .ok_or_else(|| "sparse conv fused output tile count overflow".to_string())?;
                let cube_count = calculate_cube_count_elemwise(output_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_fused_oc4_kernel::launch_unchecked::<R>(
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
        let output_elements_u32 = u32::try_from(output_elements).map_err(|_| {
            "sparse conv output size exceeds u32::MAX for split-k kernel".to_string()
        })?;
        let split_k_u32 = u32::try_from(split_k)
            .map_err(|_| "sparse conv split-k exceeds u32::MAX".to_string())?;

        match kernel_variant {
            SparseConvKernelVariant::Baseline => {
                let partial_cube_count = calculate_cube_count_elemwise(partial_elements, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.as_tensor_arg::<f32>(1),
                        neighbor_rows.as_tensor_arg::<i32>(1),
                        weight.as_tensor_arg::<f32>(1),
                        partial.as_tensor_arg::<f32>(1),
                        ScalarArg::new(config.out_channels as u32),
                        ScalarArg::new(kernel_rows as u32),
                        ScalarArg::new(config.in_channels as u32),
                        ScalarArg::new(config.in_channels_per_group as u32),
                        ScalarArg::new(config.out_channels_per_group as u32),
                        ScalarArg::new(output_elements_u32),
                        ScalarArg::new(split_k_u32),
                    );
                }
            }
            SparseConvKernelVariant::FusedOc4 => {
                let blocks_per_row = config.out_channels.div_ceil(FUSED_OC_TILE as usize) as u32;
                let partial_blocks = query_rows
                    .checked_mul(blocks_per_row as usize)
                    .and_then(|value| value.checked_mul(split_k))
                    .ok_or_else(|| "sparse conv fused split-k tile count overflow".to_string())?;
                let partial_cube_count = calculate_cube_count_elemwise(partial_blocks, cube_dim);
                unsafe {
                    sparse_subm_conv_splitk_partial_fused_oc4_kernel::launch_unchecked::<R>(
                        &input.client,
                        partial_cube_count,
                        cube_dim,
                        input.as_tensor_arg::<f32>(1),
                        neighbor_rows.as_tensor_arg::<i32>(1),
                        weight.as_tensor_arg::<f32>(1),
                        partial.as_tensor_arg::<f32>(1),
                        ScalarArg::new(config.out_channels as u32),
                        ScalarArg::new(kernel_rows as u32),
                        ScalarArg::new(config.in_channels as u32),
                        ScalarArg::new(config.in_channels_per_group as u32),
                        ScalarArg::new(config.out_channels_per_group as u32),
                        ScalarArg::new(output_elements_u32),
                        ScalarArg::new(split_k_u32),
                    );
                }
            }
        }

        let finalize_cube_count = calculate_cube_count_elemwise(output_elements, cube_dim);
        unsafe {
            sparse_subm_conv_splitk_finalize_kernel::launch_unchecked::<R>(
                &input.client,
                finalize_cube_count,
                cube_dim,
                partial.as_tensor_arg::<f32>(1),
                bias.as_tensor_arg::<f32>(1),
                output.as_tensor_arg::<f32>(1),
                ScalarArg::new(config.out_channels as u32),
                ScalarArg::new(output_elements_u32),
                ScalarArg::new(split_k_u32),
            );
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
    let output = sparse_subm_conv_forward_cubecl_impl(
        config,
        input.into_primitive().tensor(),
        neighbor_rows.into_primitive(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
        forward,
    )?;
    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
}

/// Build a device tensor containing sparse neighbor row indices.
pub fn neighbor_rows_tensor_from_coords(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let kernel_rows = kernel_rows(config)?;
    let backend = resolve_neighbor_backend(coords.len(), kernel_rows);
    let key = neighbor_cache_key(config, coords, device, backend);
    if let Some(hit) = NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        NEIGHBOR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(hit);
    }
    NEIGHBOR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);

    let tensor = match backend {
        NeighborBuildBackend::Host => {
            NEIGHBOR_BUILDS_HOST.fetch_add(1, Ordering::Relaxed);
            build_neighbor_rows_tensor_host(config, coords, device)?
        }
        NeighborBuildBackend::Device => {
            NEIGHBOR_BUILDS_DEVICE.fetch_add(1, Ordering::Relaxed);
            build_neighbor_rows_tensor_device(config, coords, device)?
        }
    };

    NEIGHBOR_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(key, tensor.clone());
        trim_cache(&mut cache);
    });
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

    let input_rows = input.shape.dims[0];
    let query_rows = neighbor_rows.shape.dims[0];
    if input.shape.dims[1] != config.in_channels {
        return Err(format!(
            "sparse conv input channel mismatch: got {} expected {}",
            input.shape.dims[1], config.in_channels
        ));
    }
    if query_rows > input_rows {
        return Err(format!(
            "sparse conv neighbor row count exceeds input rows: got {} input rows {}",
            query_rows, input_rows
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
    use std::sync::Mutex;

    use burn::tensor::Tensor;

    use crate::{SparseSubmConvConfig, SparseSubmConvWeights, sparse_subm_conv_forward_flex};

    use super::{
        DefaultWgpuBackend, clear_neighbor_rows_tensor_cache, neighbor_rows_build_stats,
        neighbor_rows_tensor_from_coords, reset_neighbor_rows_build_stats,
        sparse_subm_conv_forward_wgpu, sparse_subm_conv_forward_wgpu_with_config,
        SparseWgpuForwardConfig, SparseWgpuKernelVariant,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

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
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let _guard = ENV_LOCK.lock().expect("env lock");
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

    #[test]
    fn neighbor_rows_cache_reuses_across_equivalent_coord_allocations() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "host");
        }
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
        let device = burn_wgpu::WgpuDevice::default();

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
        assert_eq!(stats.host_builds, 1);
        assert_eq!(stats.device_builds, 0);

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND");
        }
    }

    #[test]
    fn neighbor_rows_cache_reuses_across_channel_variants_with_same_topology() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "host");
        }
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
        let device = burn_wgpu::WgpuDevice::default();

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
        assert_eq!(stats.host_builds, 1);
        assert_eq!(stats.device_builds, 0);

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND");
        }
    }

    #[test]
    fn neighbor_rows_device_backend_matches_host_backend() {
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let device = burn_wgpu::WgpuDevice::default();

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "device");
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO", "hash");
        }
        let device_rows = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("device neighbor rows")
            .to_data();
        let device_rows = device_rows.as_slice::<i32>().expect("i32").to_vec();
        let device_stats = neighbor_rows_build_stats();
        assert_eq!(device_stats.cache_misses, 1);
        assert_eq!(device_stats.device_builds, 1);

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "host");
        }
        let host_rows = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("host neighbor rows")
            .to_data();
        let host_rows = host_rows.as_slice::<i32>().expect("i32").to_vec();
        let host_stats = neighbor_rows_build_stats();
        assert_eq!(host_stats.cache_misses, 1);
        assert_eq!(host_stats.host_builds, 1);

        assert_eq!(device_rows, host_rows);

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO");
        }
    }

    #[test]
    fn neighbor_rows_device_hash_matches_scan() {
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let device = burn_wgpu::WgpuDevice::default();

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "device");
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO", "scan");
        }
        let scan_rows = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("scan rows")
            .to_data();
        let scan_rows = scan_rows.as_slice::<i32>().expect("i32").to_vec();

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND", "device");
            std::env::set_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO", "hash");
        }
        let hash_rows = neighbor_rows_tensor_from_coords(&cfg, coords.as_slice(), &device)
            .expect("hash rows")
            .to_data();
        let hash_rows = hash_rows.as_slice::<i32>().expect("i32").to_vec();

        assert_eq!(scan_rows, hash_rows);

        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_BACKEND");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_NEIGHBOR_DEVICE_ALGO");
        }
    }

    #[test]
    fn wgpu_fused_oc4_matches_baseline_output() {
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let weight_len = cfg.out_channels
            * cfg.kernel_d
            * cfg.kernel_h
            * cfg.kernel_w
            * cfg.in_channels_per_group;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

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

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_SPLIT_K", "off");
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "baseline");
        }
        let baseline = sparse_subm_conv_forward_wgpu(
            &cfg,
            input_t.clone(),
            neighbors_t.clone(),
            weight_t.clone(),
            bias_t.clone(),
        )
        .expect("baseline kernel")
        .to_data();
        let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "fused_oc4");
        }
        let fused = sparse_subm_conv_forward_wgpu(&cfg, input_t, neighbors_t, weight_t, bias_t)
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

        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_SPLIT_K");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_KERNEL");
        }
    }

    #[test]
    fn wgpu_splitk_matches_default_kernel_output() {
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let weight_len = cfg.out_channels
            * cfg.kernel_d
            * cfg.kernel_h
            * cfg.kernel_w
            * cfg.in_channels_per_group;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

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

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "baseline");
            std::env::set_var("BURN_FLEX_GMM_WGPU_SPLIT_K", "off");
        }
        let baseline = sparse_subm_conv_forward_wgpu(
            &cfg,
            input_t.clone(),
            neighbors_t.clone(),
            weight_t.clone(),
            bias_t.clone(),
        )
        .expect("baseline kernel")
        .to_data();
        let baseline = baseline.as_slice::<f32>().expect("f32").to_vec();

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_SPLIT_K", "4");
        }
        let splitk = sparse_subm_conv_forward_wgpu(&cfg, input_t, neighbors_t, weight_t, bias_t)
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
        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_SPLIT_K");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_KERNEL");
        }
    }

    #[test]
    fn wgpu_fused_splitk_matches_baseline_output() {
        let _guard = ENV_LOCK.lock().expect("env lock");
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
        let weight_len = cfg.out_channels
            * cfg.kernel_d
            * cfg.kernel_h
            * cfg.kernel_w
            * cfg.in_channels_per_group;
        let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..cfg.out_channels).map(|_| rng.next_f32()).collect();

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

        unsafe {
            std::env::set_var("BURN_FLEX_GMM_WGPU_KERNEL", "baseline");
            std::env::set_var("BURN_FLEX_GMM_WGPU_SPLIT_K", "off");
        }
        let baseline = sparse_subm_conv_forward_wgpu(
            &cfg,
            input_t.clone(),
            neighbors_t.clone(),
            weight_t.clone(),
            bias_t.clone(),
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

        unsafe {
            std::env::remove_var("BURN_FLEX_GMM_WGPU_SPLIT_K");
            std::env::remove_var("BURN_FLEX_GMM_WGPU_KERNEL");
        }
    }
}

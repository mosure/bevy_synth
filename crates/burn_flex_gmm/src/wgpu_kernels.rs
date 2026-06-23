use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;

use burn::tensor::{
    DType, FloatDType, Int, Shape, Tensor as BurnTensor, TensorData, TensorPrimitive,
};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
use burn_cubecl::{CubeRuntime, tensor::CubeTensor};
use half::f16;

use crate::{SparseSubmConvConfig, kernel_rows};

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
struct Instant {
    started_ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        Self {
            started_ms: js_sys::Date::now(),
        }
    }

    fn elapsed(&self) -> std::time::Duration {
        let ms = (js_sys::Date::now() - self.started_ms).max(0.0);
        std::time::Duration::from_secs_f64(ms / 1000.0)
    }
}

/// Default WGPU backend type used by the tensor convenience wrappers.
pub type DefaultWgpuBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
type ModuleQkvRmsNormRopeOutput = (
    BurnTensor<DefaultWgpuBackend, 4>,
    BurnTensor<DefaultWgpuBackend, 4>,
    BurnTensor<DefaultWgpuBackend, 4>,
);

fn cast_float_tensor_if_needed<const D: usize>(
    tensor: BurnTensor<DefaultWgpuBackend, D>,
    dtype: burn::tensor::FloatDType,
) -> BurnTensor<DefaultWgpuBackend, D> {
    let tensor_dtype: burn::tensor::FloatDType = tensor.dtype().into();
    if tensor_dtype == dtype {
        tensor
    } else {
        tensor.cast(dtype)
    }
}

#[cube(launch_unchecked)]
fn bf16_round_to_f32_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let bits = u32::reinterpret(input[ABSOLUTE_POS]);
    let lsb = (bits >> 16u32) & 1u32;
    let rounded = bits + 0x7fffu32 + lsb;
    output[ABSOLUTE_POS] = f32::reinterpret(rounded & 0xffff0000u32);
}

pub fn bf16_round_to_f32_wgpu<const D: usize>(
    tensor: BurnTensor<DefaultWgpuBackend, D>,
) -> BurnTensor<DefaultWgpuBackend, D> {
    let dims = tensor.dims();
    let num_elements = dims.iter().product::<usize>();
    let tensor = cast_float_tensor_if_needed(tensor, FloatDType::F32);
    if num_elements == 0 {
        return tensor;
    }

    let input = burn_cubecl::kernel::into_contiguous(tensor.into_primitive().tensor());
    let output_bytes = num_elements
        .checked_mul(core::mem::size_of::<f32>())
        .expect("bf16 round output byte size overflow");
    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        input.meta.shape.clone(),
        input.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&input.client, num_elements, cube_dim);
    unsafe {
        bf16_round_to_f32_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &input.client,
            cube_count,
            cube_dim,
            input.clone().into_array_arg(),
            output.clone().into_array_arg(),
        );
    }
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

trait LaunchUncheckedResultExt {
    fn map_err<F>(self, f: F) -> Result<(), String>
    where
        F: FnOnce(()) -> String;
}

impl LaunchUncheckedResultExt for () {
    fn map_err<F>(self, _f: F) -> Result<(), String>
    where
        F: FnOnce(()) -> String,
    {
        Ok(())
    }
}

const DEFAULT_NEIGHBOR_CACHE_MAX: usize = 128;
const INVALID_NEIGHBOR: i32 = -1;
const HASH_SLOT_EMPTY: u32 = u32::MAX;
// Keep hash occupancy low so probe chains remain short on the device-only
// insertion path (avoids pathological decode latency).
const DEFAULT_NEIGHBOR_HASH_LOAD_FACTOR: usize = 8;
const DEFAULT_NEIGHBOR_HASH_MAX_PROBE: usize = 2048;
// Bucket-hash path uses parallel atomic-add insertion into fixed per-bucket
// slot arrays to avoid sorted-hash `sort_with_indices` overhead.
const DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP: usize = 16;
// Conservative auto gate from bounded stage benches:
// bucket-hash wins clearly at larger decode-like row counts, while ~10k rows
// can still favor sorted-hash due atomic contention/variance.
const DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K: usize = 32_768;
// Sorted-hash query performs a bounded linear scan after binary-searching the
// hash bucket start. Kernel query now early-terminates once the contiguous hash
// run ends, so this cap is mostly a correctness safety bound for collision-heavy
// tails rather than steady-state work.
// Sorted-hash query does a bounded linear scan after hash lower-bound search.
// Keep the decode-k3 path tight while preserving larger-kernel parity coverage.
const DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_SMALL_K: usize = 8;
const DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_MEDIUM_K: usize = 16;
const DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_LARGE_K: usize = 32;
const DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_SMALL_K: usize = 96_000;
const DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_MEDIUM_K: usize = 240_000;
const DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_LARGE_K: usize = 520_000;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL: usize = 16;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM: usize = 18;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM: usize = 24;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_LARGE: usize = 32;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MAX: usize = 1 << 16;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MEDIUM_MAX: usize = 1 << 18;
const DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_MEDIUM_MAX: usize = 1 << 24;
// Split-k auto scheduling is tuned from bounded stage-only runs in
// `tmp/runs/20260228T002022Z_conv_stage_w6_autosched`: pick split=2 for
// medium decode work and split=4 only for larger workloads. This is a
// conservative default until dedicated autotuning lands.
const DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT2: usize = 320_000_000;
const DEFAULT_SPARSE_WGPU_SPLIT_WORK_THRESHOLD_SPLIT4: usize = 760_000_000;
const DEFAULT_SPARSE_WGPU_SPLIT_CAP_ROWS: usize = 16_384;
// Keep fused-oc4 auto selection conservative. Current WGPU fused kernel is
// parity-safe but not consistently faster on common decode shapes.
const DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OC_GROUP: usize = 256;
const DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_ROWS: usize = 8_192;
const DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_INNER_WORK: usize = 1_024;
// High-inner-work decode-like shapes (for example rows~8k with in/out>=256)
// currently benchmark faster on baseline kernels. Keep auto-fused limited to
// moderate in-channel workloads and rely on explicit override for experiments.
const DEFAULT_SPARSE_WGPU_FUSED_AUTO_MAX_IN_CHANNELS_PER_GROUP: usize = 128;
// Keep borderline decode-like shapes (8192x64->256) on baseline by default.
// Focused phase evidence (tmp/runs/20260228T010347Z_sparseconv_wgpu_w6_focus_8192_oc256)
// showed baseline split-2 edges fused split-2 on p50.
const DEFAULT_SPARSE_WGPU_FUSED_AUTO_MIN_OUTPUT_WORK: usize = 2_300_000;
// Hot-shape route from W6 matrix bench
// (tmp/runs/20260228T004120Z_w6_phase_matrix_v1):
// rows=4096, single-group, inner_work<=2048 benefits from fused-oc4.
const DEFAULT_SPARSE_WGPU_FUSED_HOT_ROWS: usize = 4_096;
const DEFAULT_SPARSE_WGPU_FUSED_HOT_MAX_INNER_WORK: usize = 2_048;
const DEFAULT_SPARSE_WGPU_FUSED_HOT_MIN_OUTPUT_WORK: usize = 500_000;
const DEFAULT_SPARSE_WGPU_FUSED_HOT_MAX_OC_GROUP: usize = 128;
// Decode-like high-OC (>=256) shapes at rows>=8192 are faster without split-k
// on current kernels due partial-buffer/finalize overhead.
const DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_ROWS: usize = 8_192;
const DEFAULT_SPARSE_WGPU_SPLIT_CAP_HIGH_OC_GROUP: usize = 256;
// Mid-row very-high-OC decode convs (e.g. rows 4k-8k @ oc>=512) are also
// consistently faster in single-pass mode on current kernels. This avoids
// split-k partial/finalize overhead in the dominant decoder hotspot band.
const DEFAULT_SPARSE_WGPU_SPLIT_CAP_VERY_HIGH_OC_MIN_ROWS: usize = 4_096;
const DEFAULT_SPARSE_WGPU_SPLIT_CAP_VERY_HIGH_OC_GROUP: usize = 512;
const LAYER_NORM_STATS_PARTIAL_CHUNK: usize = 256;
const LAYER_NORM_STATS_PARTIAL_MIN_CHANNELS: usize = 1_024;
const LAYER_NORM_STATS_PARTIAL_MIN_ROWS: usize = 1_024;
const FUSED_OC_TILE: usize = 4;
const HASH_BUILD_STAT_FAIL_ROWS: usize = 0;
const HASH_BUILD_STAT_TOTAL_PROBES: usize = 1;
const HASH_BUILD_STAT_MAX_PROBE: usize = 2;
const HASH_BUILD_STAT_LEN: usize = 3;

static NEIGHBOR_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_BUILDS_HOST: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_BUILDS_DEVICE: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_SCAN_BUILDS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_BUILDS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_SCAN_BUILD_NS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_BUILD_NS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_ROWS: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_PROBE_TOTAL: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_PROBE_MAX: AtomicU64 = AtomicU64::new(0);
static NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_CALLS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_SPLITK_CALLS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_FUSED_CALLS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_TOTAL_DISPATCHES: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_TOTAL_ROWS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static SPARSE_WGPU_CONV_TOTAL_NS: AtomicU64 = AtomicU64::new(0);

fn layer_norm_partial_stats_enabled(rows: usize, channels: usize) -> bool {
    channels >= LAYER_NORM_STATS_PARTIAL_MIN_CHANNELS && rows >= LAYER_NORM_STATS_PARTIAL_MIN_ROWS
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NeighborRowsBuildStats {
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub host_builds: u64,
    pub device_builds: u64,
    pub device_scan_builds: u64,
    pub device_hash_builds: u64,
    pub device_scan_build_ns: u64,
    pub device_hash_build_ns: u64,
    pub device_hash_rows: u64,
    pub device_hash_probe_total: u64,
    pub device_hash_probe_max: u64,
    pub device_hash_insert_fail_rows: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SparseWgpuKernelStats {
    pub calls: u64,
    pub splitk_calls: u64,
    pub fused_variant_calls: u64,
    pub single_group_specialized_calls: u64,
    pub total_dispatches: u64,
    pub total_rows: u64,
    pub total_output_elements: u64,
    pub total_elapsed_ns: u64,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum NeighborBuildBackend {
    Device,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NeighborDeviceAlgo {
    Scan,
    Hash,
    SortedHash,
    BucketHash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseConvKernelVariant {
    Baseline,
    FusedOc4,
    BaselineSingleGroup,
    FusedOc4SingleGroup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseWgpuKernelVariant {
    Auto,
    Baseline,
    FusedOc4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NeighborDeviceAlgoPreference {
    Auto,
    Scan,
    SortedHash,
    HashTableSerial,
    BucketHash,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseWgpuForwardConfig {
    pub kernel_variant: SparseWgpuKernelVariant,
    pub split_k: Option<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SparseWgpuResolvedForwardConfig {
    pub kernel_variant: SparseWgpuKernelVariant,
    pub split_k: usize,
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

#[allow(clippy::useless_conversion)]
#[cube(launch_unchecked)]
fn sparse_subm_conv_kernel(
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
fn sparse_subm_conv_single_group_kernel(
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
fn sparse_subm_conv_fused_oc4_kernel(
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
fn sparse_subm_conv_single_group_fused_oc4_kernel(
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
fn sparse_subm_conv_splitk_partial_kernel(
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
fn sparse_subm_conv_splitk_partial_single_group_kernel(
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
fn sparse_subm_conv_splitk_partial_fused_oc4_kernel(
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
fn sparse_subm_conv_splitk_partial_single_group_fused_oc4_kernel(
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
fn sparse_subm_conv_splitk_finalize_kernel(
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

#[allow(clippy::eq_op)]
#[cube(launch_unchecked)]
fn neighbor_rows_from_coords_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    rows: &usize,
    kernel_rows: &usize,
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

    let mut found = batch - batch - 1;
    for in_row in 0..*rows {
        let src = in_row * 4;
        let same = coords[src] == batch
            && coords[src + 1] == nx
            && coords[src + 2] == ny
            && coords[src + 3] == nz;
        if same && found == INVALID_NEIGHBOR {
            found = i32::cast_from(in_row);
        }
    }

    if nx < 0 || ny < 0 || nz < 0 {
        found = INVALID_NEIGHBOR;
    }
    neighbor_rows[out_idx] = found;
}

#[cube]
fn spatial_hash_u32(batch: i32, x: i32, y: i32, z: i32) -> usize {
    let b = usize::cast_from(batch);
    let xx = usize::cast_from(x);
    let yy = usize::cast_from(y);
    let zz = usize::cast_from(z);
    let mut hash = b * 0x9e37_79b1usize;
    hash ^= xx * 0x85eb_ca77usize;
    hash ^= yy * 0xc2b2_ae3dusize;
    hash ^= zz * 0x27d4_eb2fusize;
    hash
}

#[cube(launch_unchecked)]
fn neighbor_coord_hash_kernel(coords: &Array<i32>, hashes: &mut Array<i32>, rows: &usize) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }
    let row = ABSOLUTE_POS;
    let coord_base = row * 4;
    let batch = coords[coord_base];
    let x = coords[coord_base + 1];
    let y = coords[coord_base + 2];
    let z = coords[coord_base + 3];
    hashes[row] = i32::cast_from(spatial_hash_u32(batch, x, y, z));
}

macro_rules! define_neighbor_rows_from_sorted_hash_kernel {
    ($name:ident, $binary_steps:expr) => {
        #[cube(launch_unchecked)]
        fn $name(
            coords: &Array<i32>,
            offsets: &Array<i32>,
            sorted_hashes: &Array<i32>,
            sorted_rows: &Array<i32>,
            neighbor_rows: &mut Array<i32>,
            rows: &usize,
            kernel_rows: &usize,
            max_match_scan: &usize,
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

            let query_hash = i32::cast_from(spatial_hash_u32(batch, nx, ny, nz));
            let lo = RuntimeCell::<usize>::new(0);
            let hi = RuntimeCell::<usize>::new(*rows);
            for _ in 0..$binary_steps {
                let lo_v = lo.read();
                let hi_v = hi.read();
                if lo_v < hi_v {
                    let mid = lo_v + (hi_v - lo_v) / 2;
                    let mid_hash = sorted_hashes[mid];
                    if mid_hash < query_hash {
                        lo.store(mid + 1);
                    } else {
                        hi.store(mid);
                    }
                }
            }

            let start = lo.read();
            if start >= *rows || sorted_hashes[start] != query_hash {
                neighbor_rows[out_idx] = INVALID_NEIGHBOR;
                terminate!();
            }

            let best = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);
            let active = RuntimeCell::<i32>::new(1);
            for scan in 0..*max_match_scan {
                if active.read() == 1 {
                    let idx = start + scan;
                    if idx < *rows {
                        if sorted_hashes[idx] == query_hash {
                            let candidate = sorted_rows[idx];
                            if candidate >= 0 {
                                let candidate_base = usize::cast_from(candidate) * 4;
                                let same = coords[candidate_base] == batch
                                    && coords[candidate_base + 1] == nx
                                    && coords[candidate_base + 2] == ny
                                    && coords[candidate_base + 3] == nz;
                                if same {
                                    let prev = best.read();
                                    if prev == INVALID_NEIGHBOR || candidate < prev {
                                        best.store(candidate);
                                    }
                                }
                            }
                        } else {
                            // Sorted hashes are contiguous by key; once the run
                            // ends we can stop scanning without losing matches.
                            active.store(0);
                        }
                    } else {
                        active.store(0);
                    }
                }
            }

            neighbor_rows[out_idx] = best.read();
        }
    };
}

// Keep binary-search loop bounds compile-time static. Runtime-gated loop steps
// regressed sorted-hash parity on current CubeCL/WGSL lowering.
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_16, 16);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_18, 18);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_24, 24);
define_neighbor_rows_from_sorted_hash_kernel!(neighbor_rows_from_sorted_hash_kernel_32, 32);

#[cube(launch_unchecked)]
fn neighbor_hash_reset_kernel(table_rows: &mut Array<u32>, fill: &u32) {
    if ABSOLUTE_POS >= table_rows.len() {
        terminate!();
    }
    table_rows[ABSOLUTE_POS] = *fill;
}

#[cube(launch_unchecked)]
fn neighbor_hash_stats_reset_kernel(stats: &mut Array<i32>, fill: &i32) {
    if ABSOLUTE_POS >= stats.len() {
        terminate!();
    }
    stats[ABSOLUTE_POS] = *fill;
}

#[cube(launch_unchecked)]
fn neighbor_hash_build_serial_kernel(
    coords: &Array<i32>,
    table_rows: &mut Array<u32>,
    table_coords: &mut Array<i32>,
    build_stats: &mut Array<i32>,
    rows: &usize,
    table_mask: &usize,
    max_probe: &usize,
) {
    // NOTE: compare_exchange atomics currently panic on cubecl-spirv for this
    // path ("Atomic should have a scope registered"), so we keep a deterministic
    // single-lane device-side insertion kernel until upstream atomic-scope
    // support is fixed.
    if ABSOLUTE_POS != 0 {
        terminate!();
    }

    let total_probes = RuntimeCell::<i32>::new(0);
    let max_probe_seen = RuntimeCell::<i32>::new(0);
    let fail_rows = RuntimeCell::<i32>::new(0);

    for row in 0..*rows {
        let coord_base = row * 4;
        let batch = coords[coord_base];
        let x = coords[coord_base + 1];
        let y = coords[coord_base + 2];
        let z = coords[coord_base + 3];
        let hash = spatial_hash_u32(batch, x, y, z);
        let row_u32 = u32::cast_from(row);
        let inserted = RuntimeCell::<i32>::new(0);
        let row_probe_steps = RuntimeCell::<i32>::new(0);

        for probe in 0..*max_probe {
            if inserted.read() == 0 {
                let slot = (hash + probe) & *table_mask;
                let slot_state = table_rows[slot];
                if slot_state == HASH_SLOT_EMPTY {
                    let dst = slot * 4;
                    table_coords[dst] = batch;
                    table_coords[dst + 1] = x;
                    table_coords[dst + 2] = y;
                    table_coords[dst + 3] = z;
                    table_rows[slot] = row_u32;
                    row_probe_steps.store(i32::cast_from(probe + 1));
                    inserted.store(1);
                } else {
                    let dst = slot * 4;
                    let same = table_coords[dst] == batch
                        && table_coords[dst + 1] == x
                        && table_coords[dst + 2] == y
                        && table_coords[dst + 3] == z;
                    if same {
                        // Duplicate coords can appear in malformed inputs; keep
                        // query semantics deterministic by retaining lowest row.
                        let current = table_rows[slot];
                        table_rows[slot] = current.min(row_u32);
                        row_probe_steps.store(i32::cast_from(probe + 1));
                        inserted.store(1);
                    }
                }
            }
        }

        if inserted.read() == 0 {
            fail_rows.store(fail_rows.read() + 1);
            let max_probe_i32 = i32::cast_from(*max_probe);
            total_probes.store(total_probes.read() + max_probe_i32);
            max_probe_seen.store(max_probe_seen.read().max(max_probe_i32));
        } else {
            let used = row_probe_steps.read();
            total_probes.store(total_probes.read() + used);
            max_probe_seen.store(max_probe_seen.read().max(used));
        }
    }

    build_stats[HASH_BUILD_STAT_FAIL_ROWS] = fail_rows.read();
    build_stats[HASH_BUILD_STAT_TOTAL_PROBES] = total_probes.read();
    build_stats[HASH_BUILD_STAT_MAX_PROBE] = max_probe_seen.read();
}

#[cube(launch_unchecked)]
fn neighbor_hash_query_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    table_rows: &Array<u32>,
    table_coords: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    kernel_rows: &usize,
    table_mask: &usize,
    max_probe: &usize,
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
                    found.store(i32::cast_from(state));
                    active.store(0);
                }
            }
        }
    }

    neighbor_rows[out_idx] = found.read();
}

#[cube(launch_unchecked)]
fn neighbor_bucket_hash_build_kernel(
    coords: &Array<i32>,
    bucket_counts: &mut Array<Atomic<u32>>,
    bucket_rows: &mut Array<i32>,
    overflow_rows: &mut Array<Atomic<i32>>,
    rows: &usize,
    bucket_mask: &usize,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let coord_base = row * 4;
    let batch = coords[coord_base];
    let x = coords[coord_base + 1];
    let y = coords[coord_base + 2];
    let z = coords[coord_base + 3];
    let hash = spatial_hash_u32(batch, x, y, z);
    let bucket = hash & *bucket_mask;
    let slot = usize::cast_from(bucket_counts[bucket].fetch_add(u32::cast_from(1usize)));

    if slot < DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP {
        let dst = bucket * DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP + slot;
        bucket_rows[dst] = i32::cast_from(row);
    } else {
        overflow_rows[0].fetch_add(i32::cast_from(1usize));
    }
}

#[cube(launch_unchecked)]
fn neighbor_bucket_hash_query_kernel(
    coords: &Array<i32>,
    offsets: &Array<i32>,
    bucket_counts: &Array<u32>,
    bucket_rows: &Array<i32>,
    neighbor_rows: &mut Array<i32>,
    rows: &usize,
    kernel_rows: &usize,
    bucket_mask: &usize,
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
    let bucket = hash & *bucket_mask;
    let bucket_count =
        usize::cast_from(bucket_counts[bucket]).min(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP);
    let best = RuntimeCell::<i32>::new(INVALID_NEIGHBOR);

    for slot in 0..DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP {
        if slot < bucket_count {
            let candidate = bucket_rows[bucket * DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP + slot];
            if candidate >= 0 {
                let candidate_row = usize::cast_from(candidate);
                if candidate_row < *rows {
                    let base = candidate_row * 4;
                    let same = coords[base] == batch
                        && coords[base + 1] == nx
                        && coords[base + 2] == ny
                        && coords[base + 3] == nz;
                    if same {
                        let prev = best.read();
                        if prev == INVALID_NEIGHBOR || candidate < prev {
                            best.store(candidate);
                        }
                    }
                }
            }
        }
    }

    neighbor_rows[out_idx] = best.read();
}

#[cube(launch_unchecked)]
fn dense_trilinear_sample_attrs_kernel(
    positions: &Array<f32>,
    occupancy: &Array<i32>,
    attrs: &Array<f32>,
    output: &mut Array<f32>,
    rows: &usize,
    dim_x: &f32,
    dim_y: &f32,
    dim_z: &f32,
    max_x: &i32,
    max_y: &i32,
    max_z: &i32,
    max_x_f: &f32,
    max_y_f: &f32,
    max_z_f: &f32,
    stride_x: &i32,
    stride_xy: &i32,
) {
    if ABSOLUTE_POS >= *rows {
        terminate!();
    }

    let row = ABSOLUTE_POS;
    let pos_base = row * 3;
    let mut cx = (positions[pos_base] + 0.5) * *dim_x;
    let mut cy = (positions[pos_base + 1] + 0.5) * *dim_y;
    let mut cz = (positions[pos_base + 2] + 0.5) * *dim_z;
    cx = cx.max(0.0).min(*max_x_f);
    cy = cy.max(0.0).min(*max_y_f);
    cz = cz.max(0.0).min(*max_z_f);

    let x0 = i32::cast_from(cx).clamp(0, *max_x);
    let y0 = i32::cast_from(cy).clamp(0, *max_y);
    let z0 = i32::cast_from(cz).clamp(0, *max_z);
    let x1 = (x0 + 1).clamp(0, *max_x);
    let y1 = (y0 + 1).clamp(0, *max_y);
    let z1 = (z0 + 1).clamp(0, *max_z);

    let fx = cx - f32::cast_from(x0);
    let fy = cy - f32::cast_from(y0);
    let fz = cz - f32::cast_from(z0);
    let wx0 = 1.0 - fx;
    let wy0 = 1.0 - fy;
    let wz0 = 1.0 - fz;
    let wx1 = fx;
    let wy1 = fy;
    let wz1 = fz;

    let mut a0 = 0.0;
    let mut a1 = 0.0;
    let mut a2 = 0.0;
    let mut a3 = 0.0;
    let mut a4 = 0.0;
    let mut a5 = 0.0;
    let mut wsum = 0.0;

    let w000 = wx0 * wy0 * wz0;
    if w000 > 0.0 {
        let idx000 = usize::cast_from((z0 * *stride_xy + y0 * *stride_x + x0).max(0));
        if occupancy[idx000] > 0 {
            let base = idx000 * 6;
            a0 += attrs[base] * w000;
            a1 += attrs[base + 1] * w000;
            a2 += attrs[base + 2] * w000;
            a3 += attrs[base + 3] * w000;
            a4 += attrs[base + 4] * w000;
            a5 += attrs[base + 5] * w000;
            wsum += w000;
        }
    }
    let w100 = wx1 * wy0 * wz0;
    if w100 > 0.0 {
        let idx100 = usize::cast_from((z0 * *stride_xy + y0 * *stride_x + x1).max(0));
        if occupancy[idx100] > 0 {
            let base = idx100 * 6;
            a0 += attrs[base] * w100;
            a1 += attrs[base + 1] * w100;
            a2 += attrs[base + 2] * w100;
            a3 += attrs[base + 3] * w100;
            a4 += attrs[base + 4] * w100;
            a5 += attrs[base + 5] * w100;
            wsum += w100;
        }
    }
    let w010 = wx0 * wy1 * wz0;
    if w010 > 0.0 {
        let idx010 = usize::cast_from((z0 * *stride_xy + y1 * *stride_x + x0).max(0));
        if occupancy[idx010] > 0 {
            let base = idx010 * 6;
            a0 += attrs[base] * w010;
            a1 += attrs[base + 1] * w010;
            a2 += attrs[base + 2] * w010;
            a3 += attrs[base + 3] * w010;
            a4 += attrs[base + 4] * w010;
            a5 += attrs[base + 5] * w010;
            wsum += w010;
        }
    }
    let w110 = wx1 * wy1 * wz0;
    if w110 > 0.0 {
        let idx110 = usize::cast_from((z0 * *stride_xy + y1 * *stride_x + x1).max(0));
        if occupancy[idx110] > 0 {
            let base = idx110 * 6;
            a0 += attrs[base] * w110;
            a1 += attrs[base + 1] * w110;
            a2 += attrs[base + 2] * w110;
            a3 += attrs[base + 3] * w110;
            a4 += attrs[base + 4] * w110;
            a5 += attrs[base + 5] * w110;
            wsum += w110;
        }
    }
    let w001 = wx0 * wy0 * wz1;
    if w001 > 0.0 {
        let idx001 = usize::cast_from((z1 * *stride_xy + y0 * *stride_x + x0).max(0));
        if occupancy[idx001] > 0 {
            let base = idx001 * 6;
            a0 += attrs[base] * w001;
            a1 += attrs[base + 1] * w001;
            a2 += attrs[base + 2] * w001;
            a3 += attrs[base + 3] * w001;
            a4 += attrs[base + 4] * w001;
            a5 += attrs[base + 5] * w001;
            wsum += w001;
        }
    }
    let w101 = wx1 * wy0 * wz1;
    if w101 > 0.0 {
        let idx101 = usize::cast_from((z1 * *stride_xy + y0 * *stride_x + x1).max(0));
        if occupancy[idx101] > 0 {
            let base = idx101 * 6;
            a0 += attrs[base] * w101;
            a1 += attrs[base + 1] * w101;
            a2 += attrs[base + 2] * w101;
            a3 += attrs[base + 3] * w101;
            a4 += attrs[base + 4] * w101;
            a5 += attrs[base + 5] * w101;
            wsum += w101;
        }
    }
    let w011 = wx0 * wy1 * wz1;
    if w011 > 0.0 {
        let idx011 = usize::cast_from((z1 * *stride_xy + y1 * *stride_x + x0).max(0));
        if occupancy[idx011] > 0 {
            let base = idx011 * 6;
            a0 += attrs[base] * w011;
            a1 += attrs[base + 1] * w011;
            a2 += attrs[base + 2] * w011;
            a3 += attrs[base + 3] * w011;
            a4 += attrs[base + 4] * w011;
            a5 += attrs[base + 5] * w011;
            wsum += w011;
        }
    }
    let w111 = wx1 * wy1 * wz1;
    if w111 > 0.0 {
        let idx111 = usize::cast_from((z1 * *stride_xy + y1 * *stride_x + x1).max(0));
        if occupancy[idx111] > 0 {
            let base = idx111 * 6;
            a0 += attrs[base] * w111;
            a1 += attrs[base + 1] * w111;
            a2 += attrs[base + 2] * w111;
            a3 += attrs[base + 3] * w111;
            a4 += attrs[base + 4] * w111;
            a5 += attrs[base + 5] * w111;
            wsum += w111;
        }
    }

    let out_base = row * 7;
    if wsum > 1.0e-8 {
        let inv = 1.0 / wsum;
        output[out_base] = a0 * inv;
        output[out_base + 1] = a1 * inv;
        output[out_base + 2] = a2 * inv;
        output[out_base + 3] = a3 * inv;
        output[out_base + 4] = a4 * inv;
        output[out_base + 5] = a5 * inv;
        output[out_base + 6] = wsum;
    } else {
        output[out_base] = 0.0;
        output[out_base + 1] = 0.0;
        output[out_base + 2] = 0.0;
        output[out_base + 3] = 0.0;
        output[out_base + 4] = 0.0;
        output[out_base + 5] = 0.0;
        output[out_base + 6] = 0.0;
    }
}

fn resolve_cube_dim() -> CubeDim {
    CubeDim::new_1d(256)
}

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
fn layer_norm_row_stats_debug_wgpu(
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

pub fn dense_trilinear_sample_attrs_wgpu(
    positions: BurnTensor<DefaultWgpuBackend, 2>,
    occupancy: BurnTensor<DefaultWgpuBackend, 1, Int>,
    attrs: BurnTensor<DefaultWgpuBackend, 2>,
    spatial: [usize; 3],
) -> Result<BurnTensor<DefaultWgpuBackend, 2>, String> {
    let [rows, pos_cols] = positions.dims();
    if pos_cols != 3 {
        return Err(format!(
            "dense trilinear positions tensor must have 3 columns, got {pos_cols}"
        ));
    }
    let [cells, attr_cols] = attrs.dims();
    if attr_cols != 6 {
        return Err(format!(
            "dense trilinear attrs tensor must have 6 columns, got {attr_cols}"
        ));
    }
    let [occupancy_len] = occupancy.dims();
    let expected_cells = spatial[0]
        .checked_mul(spatial[1])
        .and_then(|value| value.checked_mul(spatial[2]))
        .ok_or_else(|| {
            format!(
                "dense trilinear spatial volume overflow: [{},{},{}]",
                spatial[0], spatial[1], spatial[2]
            )
        })?;
    if expected_cells == 0 {
        return Err("dense trilinear sampling requires non-empty spatial volume".to_string());
    }
    if cells != expected_cells || occupancy_len != expected_cells {
        return Err(format!(
            "dense trilinear tensor length mismatch: attrs_rows={} occupancy_len={} expected_cells={}",
            cells, occupancy_len, expected_cells
        ));
    }
    if rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2>::zeros(
            [0, 7],
            &positions.device(),
        ));
    }

    let max_x = i32::try_from(spatial[0].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial x={} exceeds i32 range", spatial[0]))?;
    let max_y = i32::try_from(spatial[1].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial y={} exceeds i32 range", spatial[1]))?;
    let max_z = i32::try_from(spatial[2].saturating_sub(1))
        .map_err(|_| format!("dense trilinear spatial z={} exceeds i32 range", spatial[2]))?;
    let stride_x = i32::try_from(spatial[0])
        .map_err(|_| format!("dense trilinear spatial x={} exceeds i32 range", spatial[0]))?;
    let stride_xy_u64 = (spatial[0] as u64)
        .checked_mul(spatial[1] as u64)
        .ok_or_else(|| {
            format!(
                "dense trilinear stride overflow for spatial=[{},{},{}]",
                spatial[0], spatial[1], spatial[2]
            )
        })?;
    let stride_xy = i32::try_from(stride_xy_u64).map_err(|_| {
        format!(
            "dense trilinear stride_xy={} exceeds i32 range",
            stride_xy_u64
        )
    })?;

    let positions_p = positions.reshape([rows * 3]).into_primitive().tensor();
    let occupancy_p = occupancy.into_primitive();
    let attrs_p = attrs.reshape([cells * 6]).into_primitive().tensor();
    let output_elements = rows
        .checked_mul(7)
        .ok_or_else(|| "dense trilinear output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| "dense trilinear output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        positions_p.client.clone(),
        positions_p.device.clone(),
        Shape::new([rows, 7]),
        positions_p.client.empty(output_bytes),
        DType::F32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&positions_p.client, rows, cube_dim);
    let dim_x = spatial[0] as f32;
    let dim_y = spatial[1] as f32;
    let dim_z = spatial[2] as f32;
    let max_x_f = max_x as f32;
    let max_y_f = max_y as f32;
    let max_z_f = max_z as f32;
    unsafe {
        dense_trilinear_sample_attrs_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &positions_p.client,
            cube_count,
            cube_dim,
            positions_p.clone().into_array_arg(),
            occupancy_p.clone().into_array_arg(),
            attrs_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            dim_x,
            dim_y,
            dim_z,
            max_x,
            max_y,
            max_z,
            max_x_f,
            max_y_f,
            max_z_f,
            stride_x,
            stride_xy,
        )
        .map_err(|err| format!("dense_trilinear_sample_attrs_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::from_primitive(TensorPrimitive::Float(output)))
}

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
struct ResolvedSparseWgpuForwardConfigInternal {
    kernel_variant: SparseConvKernelVariant,
    split_k: usize,
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

fn resolve_sparse_wgpu_forward_config_internal(
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

fn resolve_neighbor_backend(_rows: usize, _kernel_rows: usize) -> NeighborBuildBackend {
    // Canonical path: device-resident neighbor map generation.
    NeighborBuildBackend::Device
}

fn resolve_neighbor_device_algo(
    rows: usize,
    kernel_rows: usize,
    preference: NeighborDeviceAlgoPreference,
) -> NeighborDeviceAlgo {
    #[cfg(target_arch = "wasm32")]
    if matches!(preference, NeighborDeviceAlgoPreference::SortedHash) {
        if kernel_rows <= 64 && rows >= DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K {
            return NeighborDeviceAlgo::BucketHash;
        }
        return NeighborDeviceAlgo::Hash;
    }
    match preference {
        NeighborDeviceAlgoPreference::Auto => {}
        NeighborDeviceAlgoPreference::Scan => return NeighborDeviceAlgo::Scan,
        NeighborDeviceAlgoPreference::SortedHash => return NeighborDeviceAlgo::SortedHash,
        NeighborDeviceAlgoPreference::HashTableSerial => return NeighborDeviceAlgo::Hash,
        NeighborDeviceAlgoPreference::BucketHash => return NeighborDeviceAlgo::BucketHash,
    }
    let work = rows.saturating_mul(kernel_rows);
    // Tuned from bounded stage-only bench runs in docs/reports/parity_gap:
    // small kernels cross earlier; very large kernels need more rows before
    // sort+search amortizes launch/sort overhead over scan.
    let sorted_threshold = if kernel_rows <= 64 {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_SMALL_K
    } else if kernel_rows <= 256 {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_MEDIUM_K
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_WORK_THRESHOLD_LARGE_K
    };
    // Bucket-hash beats sorted-hash on large decode-like small-k workloads by
    // avoiding sort_with_indices overhead; keep routing conservative so
    // mid-row shapes that still favor sorted-hash remain unchanged.
    if kernel_rows <= 64
        && rows >= DEFAULT_NEIGHBOR_BUCKET_HASH_ROWS_THRESHOLD_SMALL_K
        && work >= sorted_threshold
    {
        return NeighborDeviceAlgo::BucketHash;
    }
    #[cfg(target_arch = "wasm32")]
    if work >= sorted_threshold {
        return NeighborDeviceAlgo::Hash;
    }
    if work >= sorted_threshold {
        NeighborDeviceAlgo::SortedHash
    } else {
        NeighborDeviceAlgo::Scan
    }
}

fn resolve_neighbor_hash_load_factor() -> usize {
    DEFAULT_NEIGHBOR_HASH_LOAD_FACTOR
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
    // Keep probe work bounded for the current kernel form; the loop-break form
    // triggers cubecl-opt uniformity panics in this path.
    table_size.clamp(1, DEFAULT_NEIGHBOR_HASH_MAX_PROBE)
}

fn resolve_neighbor_bucket_hash_bucket_size(rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    rows.next_power_of_two().max(64)
}

fn resolve_neighbor_sorted_hash_match_scan(rows: usize, kernel_rows: usize) -> usize {
    if rows == 0 {
        return 1;
    }
    let cap = if kernel_rows <= 64 {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_SMALL_K
    } else if kernel_rows <= 256 {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_MEDIUM_K
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_MATCH_SCAN_LARGE_K
    };
    cap.min(rows).max(1)
}

fn resolve_neighbor_sorted_hash_search_steps(rows: usize) -> usize {
    // Keep search-step routing compile-time per kernel variant. Runtime-gated
    // loop bounds regressed parity on CubeCL/WGSL in this path. Keep the
    // decode-hot mid bucket tighter (2^16..2^18 => 18) to avoid 24-step
    // over-iteration on common 512-quality row counts.
    if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL
    } else if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_SMALL_MEDIUM_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM
    } else if rows <= DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_ROWS_MEDIUM_MAX {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM
    } else {
        DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_LARGE
    }
}

fn elapsed_ns_u64(start: Instant) -> u64 {
    let nanos = start.elapsed().as_nanos();
    nanos.min(u128::from(u64::MAX)) as u64
}

fn record_neighbor_device_build(algo: NeighborDeviceAlgo, elapsed_ns: u64) {
    NEIGHBOR_BUILDS_DEVICE.fetch_add(1, Ordering::Relaxed);
    match algo {
        NeighborDeviceAlgo::Scan => {
            NEIGHBOR_DEVICE_SCAN_BUILDS.fetch_add(1, Ordering::Relaxed);
            NEIGHBOR_DEVICE_SCAN_BUILD_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
        NeighborDeviceAlgo::Hash
        | NeighborDeviceAlgo::SortedHash
        | NeighborDeviceAlgo::BucketHash => {
            NEIGHBOR_DEVICE_HASH_BUILDS.fetch_add(1, Ordering::Relaxed);
            NEIGHBOR_DEVICE_HASH_BUILD_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
        }
    }
}

fn record_sparse_wgpu_conv_call(
    rows: usize,
    output_elements: usize,
    dispatches: usize,
    split_k: usize,
    kernel_variant: SparseConvKernelVariant,
    elapsed_ns: u64,
) {
    SPARSE_WGPU_CONV_CALLS.fetch_add(1, Ordering::Relaxed);
    if split_k > 1 {
        SPARSE_WGPU_CONV_SPLITK_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if matches!(
        kernel_variant,
        SparseConvKernelVariant::FusedOc4 | SparseConvKernelVariant::FusedOc4SingleGroup
    ) {
        SPARSE_WGPU_CONV_FUSED_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    if matches!(
        kernel_variant,
        SparseConvKernelVariant::BaselineSingleGroup | SparseConvKernelVariant::FusedOc4SingleGroup
    ) {
        SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS.fetch_add(1, Ordering::Relaxed);
    }
    SPARSE_WGPU_CONV_TOTAL_DISPATCHES.fetch_add(dispatches as u64, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS.fetch_add(output_elements as u64, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_NS.fetch_add(elapsed_ns, Ordering::Relaxed);
}

fn neighbor_cache_max_entries() -> usize {
    DEFAULT_NEIGHBOR_CACHE_MAX
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

#[cfg(feature = "wgpu-kernel")]
fn hash_tensor_identity(coords_t: &BurnTensor<DefaultWgpuBackend, 2, Int>) -> u64 {
    let primitive = coords_t.clone().into_primitive();
    // Hash buffer handle metadata without host readback so tensor-native paths can
    // reuse cached neighbor tensors across repeated decode calls. CubeCL's public
    // handle debug includes mutable memory-location state after first binding, so
    // keep only the allocation id plus static view metadata.
    let memory_debug = format!("{:?}", primitive.handle.memory);
    let memory_id = stable_memory_id_from_debug(&memory_debug);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&memory_id, &mut hasher);
    std::hash::Hash::hash(&primitive.handle.offset_start, &mut hasher);
    std::hash::Hash::hash(&primitive.handle.offset_end, &mut hasher);
    std::hash::Hash::hash(&format!("{:?}", primitive.handle.stream), &mut hasher);
    std::hash::Hash::hash(&primitive.meta.shape, &mut hasher);
    std::hash::Hash::hash(&primitive.meta.strides, &mut hasher);
    std::hash::Hasher::finish(&hasher)
}

#[cfg(feature = "wgpu-kernel")]
fn stable_memory_id_from_debug(debug: &str) -> String {
    const PREFIX: &str = "ManagedMemoryId { value: ";
    let Some(start) = debug.find(PREFIX) else {
        return debug.to_string();
    };
    let rest = &debug[start + PREFIX.len()..];
    let Some(end) = rest.find('}') else {
        return debug.to_string();
    };
    rest[..end].trim().to_string()
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

#[cfg(feature = "wgpu-kernel")]
fn neighbor_cache_key_tensor(
    config: &SparseSubmConvConfig,
    coords_t: &BurnTensor<DefaultWgpuBackend, 2, Int>,
    backend: NeighborBuildBackend,
) -> NeighborRowsCacheKey {
    let [rows, _] = coords_t.dims();
    let device = coords_t.device();
    NeighborRowsCacheKey {
        config: NeighborConfigCacheKey::from(config),
        backend,
        rows,
        coords_hash: hash_tensor_identity(coords_t),
        // Keep tensor-path keys disjoint from host-path keys in shared cache map.
        device_key: format!("{device:?}:tensor"),
    }
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

fn build_neighbor_rows_tensor_device_scan(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let coords_flat = flatten_coords_i32(coords)?;
    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    )
    .reshape([rows, 4]);
    build_neighbor_rows_tensor_device_scan_tensor(config, coords_t)
}

fn build_neighbor_rows_tensor_device_scan_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let coords_p = coords_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    )
    .reshape([kernel_rows, 3]);
    let offsets_p = offsets_t.into_primitive();
    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;

    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([rows, kernel_rows]),
        coords_p.client.empty(output_bytes),
        DType::I32,
    );
    let cube_dim = resolve_cube_dim();
    let cube_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_rows_from_coords_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            cube_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            kernel_rows,
        )
        .map_err(|err| format!("neighbor_rows_from_coords_kernel launch failed: {err:?}"))?;
    }

    Ok(BurnTensor::from_primitive(output))
}

#[allow(dead_code)]
fn build_neighbor_rows_tensor_device_hash_wgsl_table(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let rows = coords.len();
    let coords_flat = flatten_coords_i32(coords)?;
    let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(coords_flat, [rows * 4]),
        device,
    )
    .reshape([rows, 4]);
    build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(config, coords_t)
}

#[allow(dead_code)]
fn build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
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

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();
    let table_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([table_size]),
        coords_p.client.empty(table_rows_bytes),
        DType::U32,
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
    let hash_build_stats = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([HASH_BUILD_STAT_LEN]),
        coords_p
            .client
            .empty(HASH_BUILD_STAT_LEN * core::mem::size_of::<i32>()),
        DType::I32,
    );
    let table_mask = table_size - 1;
    let max_probe = resolve_neighbor_hash_max_probe(table_size);

    let cube_dim = resolve_cube_dim();
    let reset_count = calculate_cube_count_elemwise(&coords_p.client, table_size, cube_dim);
    unsafe {
        neighbor_hash_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_count,
            cube_dim,
            table_rows.clone().into_array_arg(),
            HASH_SLOT_EMPTY,
        )
        .map_err(|err| format!("neighbor_hash_reset_kernel launch failed: {err:?}"))?;
    }
    let counter_reset_count =
        calculate_cube_count_elemwise(&coords_p.client, HASH_BUILD_STAT_LEN, cube_dim);
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            counter_reset_count,
            cube_dim,
            hash_build_stats.clone().into_array_arg(),
            0,
        )
        .map_err(|err| format!("neighbor_hash_stats_reset_kernel launch failed: {err:?}"))?;
    }

    let build_count = calculate_cube_count_elemwise(&coords_p.client, 1, cube_dim);
    unsafe {
        neighbor_hash_build_serial_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            build_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            table_rows.clone().into_array_arg(),
            table_coords.clone().into_array_arg(),
            hash_build_stats.clone().into_array_arg(),
            rows,
            table_mask,
            max_probe,
        )
        .map_err(|err| format!("neighbor_hash_build_serial_kernel launch failed: {err:?}"))?;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let hash_build_stats_t: BurnTensor<DefaultWgpuBackend, 1, Int> =
            BurnTensor::from_primitive(hash_build_stats);
        let hash_build_stats_data = hash_build_stats_t.to_data();
        let hash_build_stats = hash_build_stats_data
            .as_slice::<i32>()
            .map_err(|err| format!("neighbor hash build stats readback failed: {err:?}"))?;
        let fail_rows = hash_build_stats
            .get(HASH_BUILD_STAT_FAIL_ROWS)
            .copied()
            .unwrap_or(0);
        let total_probes = hash_build_stats
            .get(HASH_BUILD_STAT_TOTAL_PROBES)
            .copied()
            .unwrap_or(0)
            .max(0) as u64;
        let max_probe_used = hash_build_stats
            .get(HASH_BUILD_STAT_MAX_PROBE)
            .copied()
            .unwrap_or(0)
            .max(0) as u64;
        NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(total_probes, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(max_probe_used, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.fetch_add(fail_rows.max(0) as u64, Ordering::Relaxed);
        if fail_rows != 0 {
            return Err(format!(
                "neighbor hash build failed to insert {fail_rows} row(s); rows={rows} table_size={table_size} max_probe={max_probe} probe_total={total_probes} probe_max={max_probe_used}"
            ));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        // Browser WebGPU cannot synchronously read the diagnostic build-stat
        // tensor. Keep the wasm path device-resident and route auto selection to
        // this low-load-factor serial hash builder; native still validates the
        // failure counter above.
        let _ = hash_build_stats;
        NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
            (rows as u64).saturating_mul(max_probe as u64),
            Ordering::Relaxed,
        );
        NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(max_probe as u64, Ordering::Relaxed);
    }

    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            table_rows.clone().into_array_arg(),
            table_coords.clone().into_array_arg(),
            output.clone().into_array_arg(),
            kernel_rows,
            table_mask,
            max_probe,
        )
        .map_err(|err| format!("neighbor_hash_query_kernel launch failed: {err:?}"))?;
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
    build_neighbor_rows_tensor_device_hash_wgsl_table(config, coords, device)
}

fn build_neighbor_rows_tensor_device_sorted_hash_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let hash_bytes = rows
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor hash key byte size overflow".to_string())?;

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();

    let hash_keys = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([rows]),
        coords_p.client.empty(hash_bytes),
        DType::I32,
    );
    let cube_dim = resolve_cube_dim();
    let hash_count = calculate_cube_count_elemwise(&coords_p.client, rows, cube_dim);
    unsafe {
        neighbor_coord_hash_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            hash_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            hash_keys.clone().into_array_arg(),
            rows,
        )
        .map_err(|err| format!("neighbor_coord_hash_kernel launch failed: {err:?}"))?;
    }

    let hash_keys_t: BurnTensor<DefaultWgpuBackend, 1, Int> = BurnTensor::from_primitive(hash_keys);
    let (sorted_hashes_t, sorted_idx_t) = hash_keys_t.sort_with_indices(0);
    let sorted_hashes_p = sorted_hashes_t.into_primitive();
    let sorted_idx_p = sorted_idx_t.into_primitive();

    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );
    let search_steps = resolve_neighbor_sorted_hash_search_steps(rows);
    let match_scan = resolve_neighbor_sorted_hash_match_scan(rows, kernel_rows);
    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        match search_steps {
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL => {
                neighbor_rows_from_sorted_hash_kernel_16::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_16 launch failed: {err:?}")
                })?;
            }
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_SMALL_MEDIUM => {
                neighbor_rows_from_sorted_hash_kernel_18::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_18 launch failed: {err:?}")
                })?;
            }
            DEFAULT_NEIGHBOR_SORTED_HASH_BINARY_SEARCH_STEPS_MEDIUM => {
                neighbor_rows_from_sorted_hash_kernel_24::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_24 launch failed: {err:?}")
                })?;
            }
            _ => {
                neighbor_rows_from_sorted_hash_kernel_32::launch_unchecked::<burn_wgpu::WgpuRuntime>(
                    &coords_p.client,
                    query_count,
                    cube_dim,
                    coords_p.clone().into_array_arg(),
                    offsets_p.clone().into_array_arg(),
                    sorted_hashes_p.clone().into_array_arg(),
                    sorted_idx_p.clone().into_array_arg(),
                    output.clone().into_array_arg(),
                    rows,
                    kernel_rows,
                    match_scan,
                )
                .map_err(|err| {
                    format!("neighbor_rows_from_sorted_hash_kernel_32 launch failed: {err:?}")
                })?;
            }
        }
    }

    NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
        (output_elements as u64).saturating_mul(search_steps as u64),
        Ordering::Relaxed,
    );
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max((search_steps + match_scan) as u64, Ordering::Relaxed);

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device_bucket_hash_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let offsets = kernel_offsets(config);
    let mut offsets_flat = Vec::with_capacity(offsets.len() * 3);
    for offset in offsets {
        offsets_flat.extend_from_slice(offset.as_slice());
    }

    let output_elements = rows
        .checked_mul(kernel_rows)
        .ok_or_else(|| "neighbor row output size overflow".to_string())?;
    let output_row_bytes = output_elements
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor row output byte size overflow".to_string())?;
    let bucket_count = resolve_neighbor_bucket_hash_bucket_size(rows);
    let bucket_mask = bucket_count - 1;
    let bucket_rows_len = bucket_count
        .checked_mul(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP)
        .ok_or_else(|| "neighbor bucket-hash row table size overflow".to_string())?;
    let bucket_counts_bytes = bucket_count
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or_else(|| "neighbor bucket-hash counts byte size overflow".to_string())?;
    let bucket_rows_bytes = bucket_rows_len
        .checked_mul(core::mem::size_of::<i32>())
        .ok_or_else(|| "neighbor bucket-hash rows byte size overflow".to_string())?;

    let coords_flat_t = coords_t.reshape([rows * 4]);
    let coords_p = coords_flat_t.into_primitive();
    let offsets_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
        TensorData::new(offsets_flat, [kernel_rows * 3]),
        &coords_p.device,
    );
    let offsets_p = offsets_t.into_primitive();

    let bucket_counts = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([bucket_count]),
        coords_p.client.empty(bucket_counts_bytes),
        DType::U32,
    );
    let bucket_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([bucket_rows_len]),
        coords_p.client.empty(bucket_rows_bytes),
        DType::I32,
    );
    let overflow_rows = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([1]),
        coords_p.client.empty(core::mem::size_of::<i32>()),
        DType::I32,
    );
    let output = CubeTensor::new_contiguous(
        coords_p.client.clone(),
        coords_p.device.clone(),
        Shape::new([output_elements]),
        coords_p.client.empty(output_row_bytes),
        DType::I32,
    );

    let cube_dim = resolve_cube_dim();
    let reset_counts = calculate_cube_count_elemwise(&coords_p.client, bucket_count, cube_dim);
    unsafe {
        neighbor_hash_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_counts,
            cube_dim,
            bucket_counts.clone().into_array_arg(),
            0u32,
        )
        .map_err(|err| format!("neighbor bucket-hash counts reset launch failed: {err:?}"))?;
    }
    let reset_rows = calculate_cube_count_elemwise(&coords_p.client, bucket_rows_len, cube_dim);
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            reset_rows,
            cube_dim,
            bucket_rows.clone().into_array_arg(),
            INVALID_NEIGHBOR,
        )
        .map_err(|err| format!("neighbor bucket-hash rows reset launch failed: {err:?}"))?;
    }
    unsafe {
        neighbor_hash_stats_reset_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            calculate_cube_count_elemwise(&coords_p.client, 1, cube_dim),
            cube_dim,
            overflow_rows.clone().into_array_arg(),
            0i32,
        )
        .map_err(|err| format!("neighbor bucket-hash overflow reset launch failed: {err:?}"))?;
    }

    let build_count = calculate_cube_count_elemwise(&coords_p.client, rows, cube_dim);
    unsafe {
        neighbor_bucket_hash_build_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            build_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            bucket_counts.clone().into_array_arg(),
            bucket_rows.clone().into_array_arg(),
            overflow_rows.clone().into_array_arg(),
            rows,
            bucket_mask,
        )
        .map_err(|err| format!("neighbor_bucket_hash_build_kernel launch failed: {err:?}"))?;
    }

    let query_count = calculate_cube_count_elemwise(&coords_p.client, output_elements, cube_dim);
    unsafe {
        neighbor_bucket_hash_query_kernel::launch_unchecked::<burn_wgpu::WgpuRuntime>(
            &coords_p.client,
            query_count,
            cube_dim,
            coords_p.clone().into_array_arg(),
            offsets_p.clone().into_array_arg(),
            bucket_counts.clone().into_array_arg(),
            bucket_rows.clone().into_array_arg(),
            output.clone().into_array_arg(),
            rows,
            kernel_rows,
            bucket_mask,
        )
        .map_err(|err| format!("neighbor_bucket_hash_query_kernel launch failed: {err:?}"))?;
    }

    NEIGHBOR_DEVICE_HASH_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.fetch_add(
        (output_elements as u64).saturating_mul(DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP as u64),
        Ordering::Relaxed,
    );
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.fetch_max(
        DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP as u64,
        Ordering::Relaxed,
    );
    #[cfg(not(target_arch = "wasm32"))]
    {
        let overflow_rows_t: BurnTensor<DefaultWgpuBackend, 1, Int> =
            BurnTensor::from_primitive(overflow_rows);
        let overflow_rows_data = overflow_rows_t.to_data();
        let overflow_rows = overflow_rows_data
            .as_slice::<i32>()
            .map_err(|err| format!("neighbor bucket-hash overflow readback failed: {err:?}"))?
            .first()
            .copied()
            .unwrap_or(0)
            .max(0) as u64;

        NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.fetch_add(overflow_rows, Ordering::Relaxed);
        if overflow_rows != 0 {
            return Err(format!(
                "neighbor bucket-hash overflowed {} row(s); rows={} buckets={} slot_cap={}",
                overflow_rows, rows, bucket_count, DEFAULT_NEIGHBOR_BUCKET_HASH_SLOT_CAP
            ));
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = overflow_rows;
    }

    let neighbor_rows_1d: BurnTensor<DefaultWgpuBackend, 1, Int> =
        BurnTensor::from_primitive(output);
    Ok(neighbor_rows_1d.reshape([rows, kernel_rows]))
}

fn build_neighbor_rows_tensor_device(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    preference: NeighborDeviceAlgoPreference,
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

    let algo = resolve_neighbor_device_algo(rows, kernel_rows, preference);
    let build_start = Instant::now();
    let result = match algo {
        NeighborDeviceAlgo::Scan => build_neighbor_rows_tensor_device_scan(config, coords, device),
        NeighborDeviceAlgo::Hash => build_neighbor_rows_tensor_device_hash(config, coords, device),
        NeighborDeviceAlgo::SortedHash => {
            let coords_flat = flatten_coords_i32(coords)?;
            let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(coords_flat, [rows * 4]),
                device,
            )
            .reshape([rows, 4]);
            build_neighbor_rows_tensor_device_sorted_hash_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::BucketHash => {
            let coords_flat = flatten_coords_i32(coords)?;
            let coords_t = BurnTensor::<DefaultWgpuBackend, 1, Int>::from_data(
                TensorData::new(coords_flat, [rows * 4]),
                device,
            )
            .reshape([rows, 4]);
            build_neighbor_rows_tensor_device_bucket_hash_tensor(config, coords_t)
        }
    };
    record_neighbor_device_build(algo, elapsed_ns_u64(build_start));
    result
}

fn build_neighbor_rows_tensor_device_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    if rows == 0 || kernel_rows == 0 {
        return Ok(BurnTensor::<DefaultWgpuBackend, 2, Int>::zeros(
            [rows, kernel_rows],
            &coords_t.device(),
        ));
    }
    if rows > i32::MAX as usize {
        return Err("sparse conv row count exceeds i32::MAX for neighbor kernel".to_string());
    }

    let algo = resolve_neighbor_device_algo(rows, kernel_rows, preference);
    let build_start = Instant::now();
    let result = match algo {
        NeighborDeviceAlgo::Scan => build_neighbor_rows_tensor_device_scan_tensor(config, coords_t),
        NeighborDeviceAlgo::Hash => {
            build_neighbor_rows_tensor_device_hash_wgsl_table_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::SortedHash => {
            build_neighbor_rows_tensor_device_sorted_hash_tensor(config, coords_t)
        }
        NeighborDeviceAlgo::BucketHash => {
            build_neighbor_rows_tensor_device_bucket_hash_tensor(config, coords_t)
        }
    };
    record_neighbor_device_build(algo, elapsed_ns_u64(build_start));
    result
}

/// Build a device neighbor-row tensor directly from a device-resident coords tensor.
pub fn neighbor_rows_tensor_from_coords_tensor(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    let [rows, coord_cols] = coords_t.dims();
    if coord_cols != 4 {
        return Err(format!(
            "neighbor_rows coords tensor must have 4 columns, got {coord_cols}"
        ));
    }
    let kernel_rows = kernel_rows(config)?;
    let backend = resolve_neighbor_backend(rows, kernel_rows);
    let key = neighbor_cache_key_tensor(config, &coords_t, backend);
    if let Some(hit) = NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow().get(&key).cloned()) {
        NEIGHBOR_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return Ok(hit);
    }
    NEIGHBOR_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    let tensor = build_neighbor_rows_tensor_device_tensor(
        config,
        coords_t,
        NeighborDeviceAlgoPreference::Auto,
    )?;
    NEIGHBOR_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(key, tensor.clone());
        trim_cache(&mut cache);
    });
    Ok(tensor)
}

/// Build a device neighbor-row tensor from a device-resident coords tensor with explicit algo selection.
pub fn neighbor_rows_tensor_from_coords_tensor_with_algo(
    config: &SparseSubmConvConfig,
    coords_t: BurnTensor<DefaultWgpuBackend, 2, Int>,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    build_neighbor_rows_tensor_device_tensor(config, coords_t, preference)
}

pub fn clear_neighbor_rows_tensor_cache() {
    NEIGHBOR_TENSOR_CACHE.with(|cache| cache.borrow_mut().clear());
}

pub fn reset_neighbor_rows_build_stats() {
    NEIGHBOR_CACHE_HITS.store(0, Ordering::Relaxed);
    NEIGHBOR_CACHE_MISSES.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_HOST.store(0, Ordering::Relaxed);
    NEIGHBOR_BUILDS_DEVICE.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_SCAN_BUILDS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_BUILDS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_SCAN_BUILD_NS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_BUILD_NS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_ROWS.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_PROBE_MAX.store(0, Ordering::Relaxed);
    NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.store(0, Ordering::Relaxed);
}

pub fn neighbor_rows_build_stats() -> NeighborRowsBuildStats {
    NeighborRowsBuildStats {
        cache_hits: NEIGHBOR_CACHE_HITS.load(Ordering::Relaxed),
        cache_misses: NEIGHBOR_CACHE_MISSES.load(Ordering::Relaxed),
        host_builds: NEIGHBOR_BUILDS_HOST.load(Ordering::Relaxed),
        device_builds: NEIGHBOR_BUILDS_DEVICE.load(Ordering::Relaxed),
        device_scan_builds: NEIGHBOR_DEVICE_SCAN_BUILDS.load(Ordering::Relaxed),
        device_hash_builds: NEIGHBOR_DEVICE_HASH_BUILDS.load(Ordering::Relaxed),
        device_scan_build_ns: NEIGHBOR_DEVICE_SCAN_BUILD_NS.load(Ordering::Relaxed),
        device_hash_build_ns: NEIGHBOR_DEVICE_HASH_BUILD_NS.load(Ordering::Relaxed),
        device_hash_rows: NEIGHBOR_DEVICE_HASH_ROWS.load(Ordering::Relaxed),
        device_hash_probe_total: NEIGHBOR_DEVICE_HASH_PROBE_TOTAL.load(Ordering::Relaxed),
        device_hash_probe_max: NEIGHBOR_DEVICE_HASH_PROBE_MAX.load(Ordering::Relaxed),
        device_hash_insert_fail_rows: NEIGHBOR_DEVICE_HASH_INSERT_FAIL_ROWS.load(Ordering::Relaxed),
    }
}

pub fn reset_sparse_wgpu_kernel_stats() {
    SPARSE_WGPU_CONV_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_SPLITK_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_FUSED_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_DISPATCHES.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_ROWS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS.store(0, Ordering::Relaxed);
    SPARSE_WGPU_CONV_TOTAL_NS.store(0, Ordering::Relaxed);
}

pub fn sparse_wgpu_kernel_stats() -> SparseWgpuKernelStats {
    SparseWgpuKernelStats {
        calls: SPARSE_WGPU_CONV_CALLS.load(Ordering::Relaxed),
        splitk_calls: SPARSE_WGPU_CONV_SPLITK_CALLS.load(Ordering::Relaxed),
        fused_variant_calls: SPARSE_WGPU_CONV_FUSED_CALLS.load(Ordering::Relaxed),
        single_group_specialized_calls: SPARSE_WGPU_CONV_SINGLE_GROUP_SPECIALIZED_CALLS
            .load(Ordering::Relaxed),
        total_dispatches: SPARSE_WGPU_CONV_TOTAL_DISPATCHES.load(Ordering::Relaxed),
        total_rows: SPARSE_WGPU_CONV_TOTAL_ROWS.load(Ordering::Relaxed),
        total_output_elements: SPARSE_WGPU_CONV_TOTAL_OUTPUT_ELEMENTS.load(Ordering::Relaxed),
        total_elapsed_ns: SPARSE_WGPU_CONV_TOTAL_NS.load(Ordering::Relaxed),
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
        NeighborBuildBackend::Device => build_neighbor_rows_tensor_device(
            config,
            coords,
            device,
            NeighborDeviceAlgoPreference::Auto,
        )?,
    };

    NEIGHBOR_TENSOR_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        cache.insert(key, tensor.clone());
        trim_cache(&mut cache);
    });
    Ok(tensor)
}

/// Build a device tensor containing sparse neighbor rows with explicit algorithm selection.
pub fn neighbor_rows_tensor_from_coords_with_algo(
    config: &SparseSubmConvConfig,
    coords: &[[u32; 4]],
    device: &burn_wgpu::WgpuDevice,
    preference: NeighborDeviceAlgoPreference,
) -> Result<BurnTensor<DefaultWgpuBackend, 2, Int>, String> {
    build_neighbor_rows_tensor_device(config, coords, device, preference)
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

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use std::sync::{Mutex, MutexGuard};

    use burn::tensor::{Int, Tensor, TensorData};

    use crate::{SparseSubmConvConfig, SparseSubmConvWeights, sparse_subm_conv_forward_flex};

    use super::{
        DefaultWgpuBackend, NeighborDeviceAlgoPreference, SparseConvKernelVariant,
        SparseWgpuForwardConfig, SparseWgpuKernelVariant, build_neighbor_rows_tensor_device_scan,
        clear_neighbor_rows_tensor_cache, dense_trilinear_sample_attrs_wgpu,
        layer_norm_affine_forward_wgpu, layer_norm_affine_silu_forward_wgpu,
        layer_norm_modulated_forward_wgpu, layer_norm_row_stats_debug_wgpu,
        linear_skinny_forward_wgpu, multihead_qk_rms_norm_rope_from_qkv_coords_wgpu,
        multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu, multihead_rms_norm_forward_wgpu,
        multihead_rms_norm_module_forward_wgpu, multihead_rms_norm_rope_from_coords_wgpu,
        neighbor_rows_build_stats, neighbor_rows_tensor_from_coords,
        neighbor_rows_tensor_from_coords_tensor, neighbor_rows_tensor_from_coords_with_algo,
        reset_neighbor_rows_build_stats, reset_sparse_wgpu_kernel_stats,
        resolve_sparse_wgpu_forward_config, resolve_sparse_wgpu_forward_config_internal,
        rope_rotate_pairs_from_coords_wgpu, rope_rotate_pairs_from_phase_wgpu,
        rope_rotate_pairs_wgpu, sparse_subm_conv_forward_wgpu,
        sparse_subm_conv_forward_wgpu_im2col_matmul,
        sparse_subm_conv_forward_wgpu_im2col_matmul_fast_f16,
        sparse_subm_conv_forward_wgpu_with_config, sparse_wgpu_kernel_stats,
    };

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn env_lock_guard() -> MutexGuard<'static, ()> {
        ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
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
            ((value + 0.5) * dim).clamp(0.0, dim - 1.0)
        };
        let coord = [
            map_axis(position[0], spatial[0]),
            map_axis(position[1], spatial[1]),
            map_axis(position[2], spatial[2]),
        ];
        let base = [
            coord[0].floor() as i32,
            coord[1].floor() as i32,
            coord[2].floor() as i32,
        ];
        let frac = [
            coord[0] - base[0] as f32,
            coord[1] - base[1] as f32,
            coord[2] - base[2] as f32,
        ];

        let max_x = spatial[0].saturating_sub(1) as i32;
        let max_y = spatial[1].saturating_sub(1) as i32;
        let max_z = spatial[2].saturating_sub(1) as i32;
        let x0 = base[0].clamp(0, max_x) as usize;
        let y0 = base[1].clamp(0, max_y) as usize;
        let z0 = base[2].clamp(0, max_z) as usize;
        let x1 = (base[0] + 1).clamp(0, max_x) as usize;
        let y1 = (base[1] + 1).clamp(0, max_y) as usize;
        let z1 = (base[2] + 1).clamp(0, max_z) as usize;
        let wx0 = 1.0 - frac[0];
        let wy0 = 1.0 - frac[1];
        let wz0 = 1.0 - frac[2];
        let wx1 = frac[0];
        let wy1 = frac[1];
        let wz1 = frac[2];
        let stride_x = spatial[0];
        let stride_xy = spatial[0].saturating_mul(spatial[1]);
        let idx = |x: usize, y: usize, z: usize| -> usize { z * stride_xy + y * stride_x + x };

        let mut accum = [0.0f32; 6];
        let mut weight_sum = 0.0f32;
        let mut sample_corner = |x: usize, y: usize, z: usize, weight: f32| {
            if weight <= 0.0 {
                return;
            }
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let sampled_t =
            dense_trilinear_sample_attrs_wgpu(positions_t, occupancy_t, attrs_t, spatial)
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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

        let rotated = rope_rotate_pairs_from_coords_wgpu(x_t, coords_t, rope_freq)
            .expect("rope coords output");
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let output = linear_skinny_forward_wgpu(input_t, weight_t, bias_t)
            .expect("skinny linear kernel output");
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
                    let expected = (input[base + ch] - mean) * inv_std * (scale[mod_idx] + 1.0)
                        + shift[mod_idx];
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let reference =
            multihead_rms_norm_forward_wgpu(input_t.clone(), gamma_t.clone(), scale, eps)
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let output = multihead_rms_norm_rope_from_coords_wgpu(
            input_t, gamma_t, coords_t, rope_freq, scale, eps,
        )
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
                let even =
                    input[base + even_ch] * inv_rms * scale * gamma[head * head_dim + even_ch];
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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

        let (q_module, k_module, v_module) =
            multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu(
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();
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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
        let device = burn_wgpu::WgpuDevice::default();

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
}

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

fn resolve_cube_dim() -> CubeDim {
    CubeDim::new_1d(256)
}

fn elapsed_ns_u64(start: Instant) -> u64 {
    let nanos = start.elapsed().as_nanos();
    nanos.min(u128::from(u64::MAX)) as u64
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

mod neighbor;
mod neighbor_kernels;
mod ops;
mod sampling;
mod sampling_kernels;
mod sparse_conv;
mod sparse_conv_config;
mod sparse_conv_kernels;
#[cfg(all(test, not(target_family = "wasm")))]
mod tests;

#[cfg(test)]
use neighbor::{
    build_neighbor_rows_tensor_device_scan, resolve_neighbor_device_algo,
    resolve_neighbor_sorted_hash_search_steps,
};
#[cfg(test)]
use ops::layer_norm_row_stats_debug_wgpu;
#[cfg(test)]
use sparse_conv_config::resolve_sparse_wgpu_forward_config_internal;

pub use neighbor::{
    clear_neighbor_rows_tensor_cache, neighbor_rows_build_stats, neighbor_rows_tensor_from_coords,
    neighbor_rows_tensor_from_coords_tensor, neighbor_rows_tensor_from_coords_tensor_with_algo,
    neighbor_rows_tensor_from_coords_with_algo, reset_neighbor_rows_build_stats,
    reset_sparse_wgpu_kernel_stats, sparse_wgpu_kernel_stats,
};
pub use ops::{
    layer_norm_affine_forward_wgpu, layer_norm_affine_silu_forward_wgpu,
    layer_norm_modulated_forward_wgpu, linear_skinny_forward_wgpu,
    multihead_qk_rms_norm_rope_from_qkv_coords_wgpu,
    multihead_qkv_module_rms_norm_rope_from_qkv_coords_wgpu, multihead_rms_norm_forward_wgpu,
    multihead_rms_norm_module_forward_wgpu, multihead_rms_norm_rope_from_coords_wgpu,
    rope_rotate_pairs_from_coords_wgpu, rope_rotate_pairs_from_phase_wgpu, rope_rotate_pairs_wgpu,
};
pub use sampling::dense_trilinear_sample_attrs_wgpu;
pub use sparse_conv::{
    sparse_subm_conv_forward_cubecl, sparse_subm_conv_forward_wgpu,
    sparse_subm_conv_forward_wgpu_with_config,
};
pub use sparse_conv_config::{
    resolve_sparse_wgpu_forward_config, sparse_subm_conv_forward_wgpu_im2col_matmul,
    sparse_subm_conv_forward_wgpu_im2col_matmul_fast_f16,
};

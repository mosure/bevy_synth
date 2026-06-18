use std::{fs, path::PathBuf, time::Instant};

#[cfg(feature = "backend_wgpu")]
use burn::tensor::TensorPrimitive;
use burn::{
    backend::NdArray,
    tensor::{
        DType, Distribution, FloatDType, Shape, Tensor, TensorData, activation::softmax,
        backend::Backend, module::attention as module_attention, ops::AttentionModuleOptions,
    },
};
#[cfg(feature = "backend_wgpu")]
use burn_cubecl::kernel::attention::AttentionStrategy;
#[cfg(feature = "backend_wgpu")]
use burn_cubecl::{
    CubeRuntime,
    cubecl::{self, calculate_cube_count_elemwise, prelude::*},
    ops::numeric::empty_device_dtype,
    tensor::CubeTensor,
};
use burn_triposplat::components::{TripoSplatProfileRecord, scaled_dot_product_attention_profiled};
use clap::{Parser, ValueEnum};
#[cfg(feature = "backend_wgpu")]
use cubek::attention::routines::blackbox_accelerated::BlackboxAcceleratedStrategy;
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView},
};
use serde::Serialize;

type NdArrayBenchBackend = NdArray<f32>;
#[cfg(feature = "backend_cuda")]
type CudaBenchBackend = burn::backend::Cuda<f32, i32>;
#[cfg(feature = "backend_wgpu")]
type WgpuBenchBackend = burn::backend::Wgpu<f32, i32, u32>;
#[cfg(feature = "backend_wgpu")]
type WgpuRawBenchBackend = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(feature = "backend_wgpu")]
type WgpuF16BenchBackend = burn::backend::Wgpu<burn::tensor::f16, i32, u32>;
#[cfg(feature = "backend_wgpu")]
type WgpuFlex32BenchBackend = burn::backend::Wgpu<burn::backend::wgpu::flex32, i32, u32>;

#[derive(Debug, Parser)]
#[command(about = "Benchmark TripoSplat-shaped Burn attention calls.")]
struct Args {
    #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
    backend: BackendArg,

    #[arg(long, value_enum, default_value_t = LayoutArg::Triposplat)]
    layout: LayoutArg,

    #[arg(long, value_enum, default_value_t = AttentionModeArg::Default)]
    attention_mode: AttentionModeArg,

    #[arg(long, value_enum)]
    compare_attention_mode: Option<AttentionModeArg>,

    /// Compare one batched attention call against concatenated per-batch calls using the same mode.
    #[arg(long, default_value_t = false)]
    compare_split_batch: bool,

    #[arg(long, default_value_t = 2)]
    batch: usize,

    #[arg(long, default_value_t = 12_294)]
    query_tokens: usize,

    #[arg(long)]
    key_tokens: Option<usize>,

    #[arg(long, default_value_t = 16)]
    heads: usize,

    #[arg(long, default_value_t = 64)]
    head_dim: usize,

    #[arg(long, value_delimiter = ',', default_value = "1024,2048,full")]
    query_chunk_tokens: Vec<String>,

    #[arg(long, default_value_t = 1)]
    warmup: usize,

    #[arg(long, default_value_t = 3)]
    iters: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(long)]
    input_qkv: Option<PathBuf>,

    #[arg(long, value_enum)]
    input_layout: Option<LayoutArg>,

    #[arg(long, value_enum, default_value_t = InputDTypeArg::F32)]
    input_dtype: InputDTypeArg,

    #[arg(long, default_value = "q")]
    q_tensor: String,

    #[arg(long, default_value = "k")]
    k_tensor: String,

    #[arg(long, default_value = "v")]
    v_tensor: String,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendArg {
    Ndarray,
    Cuda,
    Wgpu,
    WgpuRaw,
    WgpuF16,
    WgpuFlex32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum LayoutArg {
    /// Exercise the TripoSplat wrapper path: [batch, tokens, heads, head_dim].
    Triposplat,
    /// Exercise Burn's public attention module directly: [batch, heads, tokens, head_dim].
    Module,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum InputDTypeArg {
    F32,
    F16,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AttentionModeArg {
    /// Use Burn's default public attention dispatch and autotuner.
    Default,
    /// Pre-scale Q by 1/sqrt(head_dim), then use Burn's default dispatch.
    PrescaledDefault,
    /// Materialize BHTD Q/K/V with public ops, then use Burn's default dispatch.
    MaterializedDefault,
    /// Set the mathematically equivalent explicit scale option, forcing Burn's full fallback path.
    ExplicitScale,
    /// Set scale=1.0, forcing Burn's fallback path without sqrt-head-dim scaling.
    ExplicitScaleOne,
    /// Force Burn's explicit-scale fallback per TripoSplat query chunk.
    ExplicitScaleChunked,
    /// Force Burn's default/autotuned module attention per TripoSplat query chunk.
    DefaultModuleChunked,
    /// Force contiguous BHTD tensors before Burn's default/autotuned module attention per chunk.
    DefaultModuleContiguousChunked,
    /// Flatten heads into the batch axis before Burn's default/autotuned module attention per chunk.
    DefaultModuleFlattenedHeadsChunked,
    /// Use contiguous default module attention for large chunks and direct primitives for tiny tails.
    DefaultModuleContiguousHybridChunked,
    /// Compose attention from public Burn tensor primitives per TripoSplat query chunk.
    DirectPrimitiveChunked,
    /// Compose attention from public Burn primitives after flattening batch and heads.
    DirectPrimitiveFlattenedHeadsChunked,
    /// Force Burn-CubeCL's multi-kernel attention fallback through the raw WGPU backend.
    CubeFallback,
    /// Force CubeK unit flash attention through Burn-CubeCL's raw WGPU backend.
    CubeFlashUnit,
    /// Force CubeK blackbox accelerated flash attention with two planes.
    CubeFlashBlackbox2,
    /// Force CubeK blackbox accelerated flash attention with four planes.
    CubeFlashBlackbox4,
    /// Force CubeK blackbox accelerated flash attention with eight planes.
    CubeFlashBlackbox8,
    /// Probe a TripoSplat-specific row-owned online-softmax WGPU flash kernel.
    WgpuFlashRow,
    /// Probe a TripoSplat-specific key-blocked online-softmax WGPU flash kernel.
    WgpuFlashBlock,
}

trait ForcedCubeAttentionBackend: Backend {
    fn forced_cube_attention_bhtd(
        mode: AttentionModeArg,
        q: Tensor<Self, 4>,
        k: Tensor<Self, 4>,
        v: Tensor<Self, 4>,
    ) -> Tensor<Self, 4>
    where
        Self: Sized,
    {
        let _ = (mode, q, k, v);
        panic!("forced CubeCL attention modes require --backend wgpu-raw");
    }
}

impl ForcedCubeAttentionBackend for NdArrayBenchBackend {}

#[cfg(feature = "backend_cuda")]
impl ForcedCubeAttentionBackend for CudaBenchBackend {}

#[cfg(feature = "backend_wgpu")]
impl ForcedCubeAttentionBackend for WgpuBenchBackend {}

#[cfg(feature = "backend_wgpu")]
impl ForcedCubeAttentionBackend for WgpuF16BenchBackend {}

#[cfg(feature = "backend_wgpu")]
impl ForcedCubeAttentionBackend for WgpuFlex32BenchBackend {}

#[cfg(feature = "backend_wgpu")]
impl ForcedCubeAttentionBackend for WgpuRawBenchBackend {
    fn forced_cube_attention_bhtd(
        mode: AttentionModeArg,
        q: Tensor<Self, 4>,
        k: Tensor<Self, 4>,
        v: Tensor<Self, 4>,
    ) -> Tensor<Self, 4> {
        let strategy = match mode {
            AttentionModeArg::CubeFallback => AttentionStrategy::Fallback,
            AttentionModeArg::CubeFlashUnit => AttentionStrategy::FlashUnit,
            AttentionModeArg::CubeFlashBlackbox2 => {
                AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                    num_planes: 2,
                    seq_q: 1,
                    seq_kv: 1,
                })
            }
            AttentionModeArg::CubeFlashBlackbox4 => {
                AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                    num_planes: 4,
                    seq_q: 1,
                    seq_kv: 1,
                })
            }
            AttentionModeArg::CubeFlashBlackbox8 => {
                AttentionStrategy::FlashBlackboxAccelerated(BlackboxAcceleratedStrategy {
                    num_planes: 8,
                    seq_q: 1,
                    seq_kv: 1,
                })
            }
            AttentionModeArg::WgpuFlashRow => {
                let out = triposplat_wgpu_row_flash_attention(
                    q.into_primitive().tensor(),
                    k.into_primitive().tensor(),
                    v.into_primitive().tensor(),
                );
                return Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(out));
            }
            AttentionModeArg::WgpuFlashBlock => {
                let out = triposplat_wgpu_block_flash_attention(
                    q.into_primitive().tensor(),
                    k.into_primitive().tensor(),
                    v.into_primitive().tensor(),
                );
                return Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(out));
            }
            _ => panic!("unsupported forced CubeCL attention mode: {mode:?}"),
        };
        let out = burn_cubecl::kernel::attention::attention::<burn_wgpu::WgpuRuntime>(
            q.into_primitive().tensor(),
            k.into_primitive().tensor(),
            v.into_primitive().tensor(),
            None,
            None,
            AttentionModuleOptions::default(),
            strategy,
        )
        .expect("forced CubeCL attention probe failed");
        Tensor::<Self, 4>::from_primitive(TensorPrimitive::Float(out))
    }
}

#[cfg(feature = "backend_wgpu")]
#[cube(launch, address_type = "dynamic")]
fn triposplat_wgpu_row_flash_attention_kernel<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    query_tokens: usize,
    key_tokens: usize,
    #[define(F)] _dtype: StorageType,
) {
    const HEAD_DIM: usize = 64;

    let row = ABSOLUTE_POS;
    let rows = out.len() / HEAD_DIM;
    if row >= rows {
        terminate!();
    }

    let batch_head = row / query_tokens;
    let query_index = row - batch_head * query_tokens;
    let q_base = (batch_head * query_tokens + query_index) * HEAD_DIM;
    let kv_base = batch_head * key_tokens * HEAD_DIM;
    let out_base = row * HEAD_DIM;
    let scale = F::new(0.125_f32);

    let mut acc = Array::<F>::new(HEAD_DIM);
    #[unroll]
    for dim in 0..HEAD_DIM {
        acc[dim] = F::new(0.0_f32);
    }

    let mut row_max = F::new(-3.4028234663852886e38_f32);
    let mut row_sum = F::new(0.0_f32);

    for key_index in 0..key_tokens {
        let key_base = kv_base + key_index * HEAD_DIM;
        let mut score = F::new(0.0_f32);

        #[unroll]
        for dim in 0..HEAD_DIM {
            score += q[q_base + dim] * k[key_base + dim];
        }
        score *= scale;

        let new_max = if score > row_max { score } else { row_max };
        let old_scale = (row_max - new_max).exp();
        let score_scale = (score - new_max).exp();
        row_sum = row_sum * old_scale + score_scale;

        #[unroll]
        for dim in 0..HEAD_DIM {
            acc[dim] = acc[dim] * old_scale + score_scale * v[key_base + dim];
        }

        row_max = new_max;
    }

    let inv_sum = F::new(1.0_f32) / row_sum;
    #[unroll]
    for dim in 0..HEAD_DIM {
        out[out_base + dim] = acc[dim] * inv_sum;
    }
}

#[cfg(feature = "backend_wgpu")]
fn triposplat_wgpu_row_flash_attention<R: CubeRuntime>(
    q: CubeTensor<R>,
    k: CubeTensor<R>,
    v: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, query_tokens, head_dim] = q.meta.shape().dims();
    let [key_batch, key_heads, key_tokens, key_head_dim] = k.meta.shape().dims();
    let [value_batch, value_heads, value_tokens, value_dim] = v.meta.shape().dims();

    assert_eq!(head_dim, 64, "WgpuFlashRow currently requires head_dim=64");
    assert_eq!(
        key_head_dim, 64,
        "WgpuFlashRow currently requires key head_dim=64"
    );
    assert_eq!(
        value_dim, 64,
        "WgpuFlashRow currently requires value_dim=64"
    );
    assert_eq!([key_batch, key_heads], [batch, heads]);
    assert_eq!(
        [value_batch, value_heads, value_tokens],
        [batch, heads, key_tokens]
    );

    let out = empty_device_dtype::<R>(
        q.client.clone(),
        q.device.clone(),
        Shape::new([batch, heads, query_tokens, value_dim]),
        q.dtype,
    );

    let rows = batch.saturating_mul(heads).saturating_mul(query_tokens);
    let cube_dim = CubeDim::new(&q.client, rows);
    let cube_count = calculate_cube_count_elemwise(&q.client, rows, cube_dim);
    let address_type = [
        q.required_address_type(),
        k.required_address_type(),
        v.required_address_type(),
        out.required_address_type(),
    ]
    .into_iter()
    .max()
    .unwrap_or_default();

    triposplat_wgpu_row_flash_attention_kernel::launch::<R>(
        &out.client,
        cube_count,
        cube_dim,
        address_type,
        q.into_array_arg(),
        k.into_array_arg(),
        v.into_array_arg(),
        out.clone().into_array_arg(),
        query_tokens,
        key_tokens,
        out.dtype.into(),
    );

    out
}

#[cfg(feature = "backend_wgpu")]
#[cube(launch, address_type = "dynamic")]
fn triposplat_wgpu_block_flash_partials_kernel<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    partials: &mut Array<F>,
    query_tokens: usize,
    key_tokens: usize,
    key_blocks: usize,
    #[define(F)] _dtype: StorageType,
) {
    const HEAD_DIM: usize = 64;
    const PARTIAL_STRIDE: usize = 66;
    const KEY_BLOCK: usize = 512;

    let pos = ABSOLUTE_POS;
    let total = partials.len() / PARTIAL_STRIDE;
    if pos >= total {
        terminate!();
    }

    let row = pos / key_blocks;
    let key_block = pos - row * key_blocks;
    let batch_head = row / query_tokens;
    let query_index = row - batch_head * query_tokens;
    let q_base = (batch_head * query_tokens + query_index) * HEAD_DIM;
    let kv_base = batch_head * key_tokens * HEAD_DIM;
    let key_start = key_block * KEY_BLOCK;
    let key_end_unclamped = key_start + KEY_BLOCK;
    let key_end = if key_end_unclamped > key_tokens {
        key_tokens
    } else {
        key_end_unclamped
    };
    let partial_base = pos * PARTIAL_STRIDE;
    let scale = F::new(0.125_f32);

    let mut acc = Array::<F>::new(HEAD_DIM);
    #[unroll]
    for dim in 0..HEAD_DIM {
        acc[dim] = F::new(0.0_f32);
    }

    let mut local_max = F::new(-3.4028234663852886e38_f32);
    let mut local_sum = F::new(0.0_f32);

    for key_index in key_start..key_end {
        let key_base = kv_base + key_index * HEAD_DIM;
        let mut score = F::new(0.0_f32);

        #[unroll]
        for dim in 0..HEAD_DIM {
            score += q[q_base + dim] * k[key_base + dim];
        }
        score *= scale;

        let new_max = if score > local_max { score } else { local_max };
        let old_scale = (local_max - new_max).exp();
        let score_scale = (score - new_max).exp();
        local_sum = local_sum * old_scale + score_scale;

        #[unroll]
        for dim in 0..HEAD_DIM {
            acc[dim] = acc[dim] * old_scale + score_scale * v[key_base + dim];
        }

        local_max = new_max;
    }

    partials[partial_base] = local_max;
    partials[partial_base + 1] = local_sum;
    #[unroll]
    for dim in 0..HEAD_DIM {
        partials[partial_base + 2 + dim] = acc[dim];
    }
}

#[cfg(feature = "backend_wgpu")]
#[cube(launch, address_type = "dynamic")]
fn triposplat_wgpu_block_flash_reduce_kernel<F: Float>(
    partials: &Array<F>,
    out: &mut Array<F>,
    key_blocks: usize,
    #[define(F)] _dtype: StorageType,
) {
    const HEAD_DIM: usize = 64;
    const PARTIAL_STRIDE: usize = 66;

    let row = ABSOLUTE_POS;
    let rows = out.len() / HEAD_DIM;
    if row >= rows {
        terminate!();
    }

    let partial_row_base = row * key_blocks * PARTIAL_STRIDE;
    let mut row_max = F::new(-3.4028234663852886e38_f32);

    for key_block in 0..key_blocks {
        let partial_base = partial_row_base + key_block * PARTIAL_STRIDE;
        let local_max = partials[partial_base];
        row_max = if local_max > row_max {
            local_max
        } else {
            row_max
        };
    }

    let mut row_sum = F::new(0.0_f32);
    let mut acc = Array::<F>::new(HEAD_DIM);
    #[unroll]
    for dim in 0..HEAD_DIM {
        acc[dim] = F::new(0.0_f32);
    }

    for key_block in 0..key_blocks {
        let partial_base = partial_row_base + key_block * PARTIAL_STRIDE;
        let local_max = partials[partial_base];
        let local_sum = partials[partial_base + 1];
        let block_scale = (local_max - row_max).exp();
        row_sum += local_sum * block_scale;

        #[unroll]
        for dim in 0..HEAD_DIM {
            acc[dim] += partials[partial_base + 2 + dim] * block_scale;
        }
    }

    let inv_sum = F::new(1.0_f32) / row_sum;
    let out_base = row * HEAD_DIM;
    #[unroll]
    for dim in 0..HEAD_DIM {
        out[out_base + dim] = acc[dim] * inv_sum;
    }
}

#[cfg(feature = "backend_wgpu")]
fn triposplat_wgpu_block_flash_attention<R: CubeRuntime>(
    q: CubeTensor<R>,
    k: CubeTensor<R>,
    v: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, query_tokens, head_dim] = q.meta.shape().dims();
    let [key_batch, key_heads, key_tokens, key_head_dim] = k.meta.shape().dims();
    let [value_batch, value_heads, value_tokens, value_dim] = v.meta.shape().dims();

    assert_eq!(
        head_dim, 64,
        "WgpuFlashBlock currently requires head_dim=64"
    );
    assert_eq!(
        key_head_dim, 64,
        "WgpuFlashBlock currently requires key head_dim=64"
    );
    assert_eq!(
        value_dim, 64,
        "WgpuFlashBlock currently requires value_dim=64"
    );
    assert_eq!([key_batch, key_heads], [batch, heads]);
    assert_eq!(
        [value_batch, value_heads, value_tokens],
        [batch, heads, key_tokens]
    );

    let out = empty_device_dtype::<R>(
        q.client.clone(),
        q.device.clone(),
        Shape::new([batch, heads, query_tokens, value_dim]),
        q.dtype,
    );

    let rows = batch.saturating_mul(heads).saturating_mul(query_tokens);
    let key_blocks = key_tokens.div_ceil(512);
    let partials = empty_device_dtype::<R>(
        q.client.clone(),
        q.device.clone(),
        Shape::new([rows, key_blocks, 66]),
        q.dtype,
    );

    let partial_units = rows.saturating_mul(key_blocks);
    let partial_cube_dim = CubeDim::new(&q.client, partial_units);
    let partial_cube_count =
        calculate_cube_count_elemwise(&q.client, partial_units, partial_cube_dim);
    let partial_address_type = [
        q.required_address_type(),
        k.required_address_type(),
        v.required_address_type(),
        partials.required_address_type(),
    ]
    .into_iter()
    .max()
    .unwrap_or_default();

    triposplat_wgpu_block_flash_partials_kernel::launch::<R>(
        &partials.client,
        partial_cube_count,
        partial_cube_dim,
        partial_address_type,
        q.into_array_arg(),
        k.into_array_arg(),
        v.into_array_arg(),
        partials.clone().into_array_arg(),
        query_tokens,
        key_tokens,
        key_blocks,
        partials.dtype.into(),
    );

    let reduce_cube_dim = CubeDim::new(&out.client, rows);
    let reduce_cube_count = calculate_cube_count_elemwise(&out.client, rows, reduce_cube_dim);
    let reduce_address_type = [
        partials.required_address_type(),
        out.required_address_type(),
    ]
    .into_iter()
    .max()
    .unwrap_or_default();

    triposplat_wgpu_block_flash_reduce_kernel::launch::<R>(
        &out.client,
        reduce_cube_count,
        reduce_cube_dim,
        reduce_address_type,
        partials.into_array_arg(),
        out.clone().into_array_arg(),
        key_blocks,
        out.dtype.into(),
    );

    out
}

#[derive(Debug, Serialize)]
struct AttentionBenchReport {
    backend_type: String,
    backend_name: String,
    layout: String,
    attention_mode: String,
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    heads: usize,
    head_dim: usize,
    warmup: usize,
    iters: usize,
    seed: u64,
    results: Vec<AttentionBenchResult>,
}

#[derive(Debug, Serialize)]
struct AttentionBenchResult {
    query_chunk_tokens: String,
    raw_query_chunk_tokens: usize,
    elapsed_ms: Vec<f64>,
    mean_ms: f64,
    min_ms: f64,
    p50_ms: f64,
    p90_ms: f64,
    profile_records: Vec<TripoSplatProfileRecord>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compare_to_attention_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    compare_diff: Option<AttentionDiffSummary>,
}

#[derive(Clone, Copy, Debug)]
struct AttentionTensorDims {
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    heads: usize,
    head_dim: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct AttentionDiffSummary {
    max_abs: f64,
    mean_abs: f64,
    rms: f64,
    count_abs_gt_0_001: usize,
    count_abs_gt_0_01: usize,
    count_abs_gt_0_1: usize,
    max_abs_flat_index: Option<usize>,
    reference_at_max_abs: Option<f32>,
    candidate_at_max_abs: Option<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    match args.backend {
        BackendArg::Ndarray => run::<NdArrayBenchBackend>(&args, Default::default()),
        BackendArg::Cuda => {
            #[cfg(feature = "backend_cuda")]
            {
                run::<CudaBenchBackend>(&args, Default::default())
            }
            #[cfg(not(feature = "backend_cuda"))]
            {
                Err(
                    "triposplat_attention_bench --backend cuda requires feature backend_cuda"
                        .into(),
                )
            }
        }
        BackendArg::Wgpu => {
            #[cfg(feature = "backend_wgpu")]
            {
                run::<WgpuBenchBackend>(&args, Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_attention_bench --backend wgpu requires feature backend_wgpu"
                        .into(),
                )
            }
        }
        BackendArg::WgpuRaw => {
            #[cfg(feature = "backend_wgpu")]
            {
                run::<WgpuRawBenchBackend>(&args, Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_attention_bench --backend wgpu-raw requires feature backend_wgpu"
                        .into(),
                )
            }
        }
        BackendArg::WgpuF16 => {
            #[cfg(feature = "backend_wgpu")]
            {
                run::<WgpuF16BenchBackend>(&args, Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_attention_bench --backend wgpu-f16 requires feature backend_wgpu"
                        .into(),
                )
            }
        }
        BackendArg::WgpuFlex32 => {
            #[cfg(feature = "backend_wgpu")]
            {
                run::<WgpuFlex32BenchBackend>(&args, Default::default())
            }
            #[cfg(not(feature = "backend_wgpu"))]
            {
                Err(
                    "triposplat_attention_bench --backend wgpu-flex32 requires feature backend_wgpu"
                        .into(),
                )
            }
        }
    }
}

fn run<B: ForcedCubeAttentionBackend>(
    args: &Args,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.batch == 0 || args.query_tokens == 0 || args.heads == 0 || args.head_dim == 0 {
        return Err("batch, query_tokens, heads, and head_dim must be > 0".into());
    }
    if args.iters == 0 {
        return Err("--iters must be > 0".into());
    }
    if args.compare_attention_mode.is_some() && args.compare_split_batch {
        return Err(
            "--compare-attention-mode and --compare-split-batch are mutually exclusive".into(),
        );
    }

    B::seed(&device, args.seed);
    let (q, k, v) = if let Some(path) = &args.input_qkv {
        load_attention_tensors::<B>(
            path,
            args.input_layout.unwrap_or(args.layout),
            args.layout,
            &args.q_tensor,
            &args.k_tensor,
            &args.v_tensor,
            args.input_dtype,
            &device,
        )?
    } else {
        let key_tokens = args.key_tokens.unwrap_or(args.query_tokens);
        if key_tokens == 0 {
            return Err("--key-tokens must be > 0".into());
        }
        random_attention_tensors::<B>(args, key_tokens, &device)
    };
    let dims = attention_tensor_dims(args.layout, &q, &k, &v)?;
    if args.compare_split_batch && dims.batch < 2 {
        return Err("--compare-split-batch requires --batch >= 2".into());
    }

    let chunk_values = args
        .query_chunk_tokens
        .iter()
        .map(|raw| parse_query_chunk_tokens(raw, dims.query_tokens))
        .collect::<Result<Vec<_>, _>>()?;

    let mut results = Vec::with_capacity(chunk_values.len());
    for (chunk_label, chunk_tokens) in chunk_values {
        for _ in 0..args.warmup {
            let mut records = Vec::new();
            let out = attention_once(
                args.layout,
                args.attention_mode,
                "attention.warmup",
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut records,
            );
            let _ = out.dims();
            B::sync(&device).expect("attention bench warmup sync failed");
        }

        let mut elapsed_ms = Vec::with_capacity(args.iters);
        let mut profile_records = Vec::new();
        for iter in 0..args.iters {
            let label = format!("attention.iter_{iter:02}");
            let start = Instant::now();
            let out = attention_once(
                args.layout,
                args.attention_mode,
                &label,
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut profile_records,
            );
            let _ = out.dims();
            B::sync(&device).expect("attention bench sync failed");
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            if matches!(args.layout, LayoutArg::Module) {
                push_direct_module_profile_record::<B>(
                    &mut profile_records,
                    label,
                    dims,
                    args.attention_mode,
                    chunk_tokens,
                    elapsed,
                    &device,
                );
            }
            elapsed_ms.push(elapsed);
        }
        let (compare_to_attention_mode, compare_diff) = if args.compare_split_batch {
            let mut reference_records = Vec::new();
            let reference = attention_split_batch_reference(
                args.layout,
                args.attention_mode,
                "attention.compare_split_reference",
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut reference_records,
            )?;
            let mut candidate_records = Vec::new();
            let candidate = attention_once(
                args.layout,
                args.attention_mode,
                "attention.compare_batched_candidate",
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut candidate_records,
            );
            B::sync(&device).expect("attention split-batch compare sync failed");
            profile_records.extend(reference_records);
            profile_records.extend(candidate_records);
            (
                Some(format!("{:?}_split_batch", args.attention_mode)),
                Some(compare_attention_outputs(reference, candidate)?),
            )
        } else if let Some(reference_mode) = args.compare_attention_mode {
            let mut reference_records = Vec::new();
            let reference = attention_once(
                args.layout,
                reference_mode,
                "attention.compare_reference",
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut reference_records,
            );
            let mut candidate_records = Vec::new();
            let candidate = attention_once(
                args.layout,
                args.attention_mode,
                "attention.compare_candidate",
                q.clone(),
                k.clone(),
                v.clone(),
                dims.head_dim,
                chunk_tokens,
                &mut candidate_records,
            );
            B::sync(&device).expect("attention compare sync failed");
            profile_records.extend(reference_records);
            profile_records.extend(candidate_records);
            (
                Some(format!("{reference_mode:?}")),
                Some(compare_attention_outputs(reference, candidate)?),
            )
        } else {
            (None, None)
        };
        results.push(AttentionBenchResult {
            query_chunk_tokens: chunk_label,
            raw_query_chunk_tokens: chunk_tokens,
            mean_ms: mean(elapsed_ms.as_slice()),
            min_ms: elapsed_ms.iter().copied().reduce(f64::min).unwrap_or(0.0),
            p50_ms: percentile(elapsed_ms.clone(), 0.50),
            p90_ms: percentile(elapsed_ms.clone(), 0.90),
            elapsed_ms,
            profile_records,
            compare_to_attention_mode,
            compare_diff,
        });
    }

    let report = AttentionBenchReport {
        backend_type: std::any::type_name::<B>().to_string(),
        backend_name: B::name(&device).to_string(),
        layout: format!("{:?}", args.layout),
        attention_mode: format!("{:?}", args.attention_mode),
        batch: dims.batch,
        query_tokens: dims.query_tokens,
        key_tokens: dims.key_tokens,
        heads: dims.heads,
        head_dim: dims.head_dim,
        warmup: args.warmup,
        iters: args.iters,
        seed: args.seed,
        results,
    };
    let json = serde_json::to_string_pretty(&report)? + "\n";
    if let Some(path) = &args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, json.as_bytes())?;
        eprintln!("[triposplat_attention_bench] wrote {}", path.display());
    } else {
        print!("{json}");
    }
    Ok(())
}

fn random_attention_tensors<B: Backend>(
    args: &Args,
    key_tokens: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    match args.layout {
        LayoutArg::Triposplat => (
            Tensor::<B, 4>::random(
                [args.batch, args.query_tokens, args.heads, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            Tensor::<B, 4>::random(
                [args.batch, key_tokens, args.heads, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            Tensor::<B, 4>::random(
                [args.batch, key_tokens, args.heads, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
        ),
        LayoutArg::Module => (
            Tensor::<B, 4>::random(
                [args.batch, args.heads, args.query_tokens, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            Tensor::<B, 4>::random(
                [args.batch, args.heads, key_tokens, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
            Tensor::<B, 4>::random(
                [args.batch, args.heads, key_tokens, args.head_dim],
                Distribution::Normal(0.0, 1.0),
                device,
            ),
        ),
    }
}

fn load_attention_tensors<B: Backend>(
    path: &PathBuf,
    input_layout: LayoutArg,
    layout: LayoutArg,
    q_name: &str,
    k_name: &str,
    v_name: &str,
    input_dtype: InputDTypeArg,
    device: &B::Device,
) -> Result<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let q = read_f32_tensor_4d::<B>(&tensors, q_name, device)?;
    let k = read_f32_tensor_4d::<B>(&tensors, k_name, device)?;
    let v = read_f32_tensor_4d::<B>(&tensors, v_name, device)?;
    attention_tensor_dims(input_layout, &q, &k, &v)?;
    let (q, k, v) = if input_layout == layout {
        (q, k, v)
    } else {
        (
            q.permute([0, 2, 1, 3]),
            k.permute([0, 2, 1, 3]),
            v.permute([0, 2, 1, 3]),
        )
    };
    let (q, k, v) = match input_dtype {
        InputDTypeArg::F32 => (q, k, v),
        InputDTypeArg::F16 => (
            q.cast(FloatDType::F16),
            k.cast(FloatDType::F16),
            v.cast(FloatDType::F16),
        ),
    };
    attention_tensor_dims(layout, &q, &k, &v)?;
    Ok((q, k, v))
}

fn read_f32_tensor_4d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let view = tensors.tensor(name)?;
    if view.dtype() != Dtype::F32 {
        return Err(format!("{name} must be F32, got {:?}", view.dtype()).into());
    }
    let shape = view.shape();
    if shape.len() != 4 {
        return Err(format!("{name} must be rank 4, got shape {shape:?}").into());
    }
    let values = f32_values(&view)?;
    let len: usize = shape.iter().product();
    Ok(
        Tensor::<B, 1>::from_data(TensorData::new(values, [len]), (device, DType::F32))
            .reshape([shape[0], shape[1], shape[2], shape[3]]),
    )
}

fn f32_values(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let chunks = view.data().chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err("F32 tensor byte length is not divisible by 4".into());
    }
    Ok(chunks
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn attention_tensor_dims<B: Backend>(
    layout: LayoutArg,
    q: &Tensor<B, 4>,
    k: &Tensor<B, 4>,
    v: &Tensor<B, 4>,
) -> Result<AttentionTensorDims, Box<dyn std::error::Error>> {
    let q_dims = q.dims();
    let k_dims = k.dims();
    let v_dims = v.dims();
    let dims = match layout {
        LayoutArg::Triposplat => AttentionTensorDims {
            batch: q_dims[0],
            query_tokens: q_dims[1],
            heads: q_dims[2],
            head_dim: q_dims[3],
            key_tokens: k_dims[1],
        },
        LayoutArg::Module => AttentionTensorDims {
            batch: q_dims[0],
            heads: q_dims[1],
            query_tokens: q_dims[2],
            head_dim: q_dims[3],
            key_tokens: k_dims[2],
        },
    };
    let expected_k = match layout {
        LayoutArg::Triposplat => [dims.batch, dims.key_tokens, dims.heads, dims.head_dim],
        LayoutArg::Module => [dims.batch, dims.heads, dims.key_tokens, dims.head_dim],
    };
    if k_dims != expected_k {
        return Err(format!(
            "k shape mismatch for {layout:?}: got {k_dims:?}, expected {expected_k:?}"
        )
        .into());
    }
    if v_dims != expected_k {
        return Err(format!(
            "v shape mismatch for {layout:?}: got {v_dims:?}, expected {expected_k:?}"
        )
        .into());
    }
    Ok(dims)
}

fn attention_split_batch_reference<B: ForcedCubeAttentionBackend>(
    layout: LayoutArg,
    attention_mode: AttentionModeArg,
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let batch = q.dims()[0];
    if batch < 2 {
        return Err("split-batch reference requires batch >= 2".into());
    }

    let mut chunks = Vec::with_capacity(batch);
    match layout {
        LayoutArg::Triposplat => {
            let [_, query_tokens, heads, q_head_dim] = q.dims();
            let key_tokens = k.dims()[1];
            let k_head_dim = k.dims()[3];
            let value_dim = v.dims()[3];
            for batch_index in 0..batch {
                let q_chunk = q.clone().slice([
                    batch_index..batch_index + 1,
                    0..query_tokens,
                    0..heads,
                    0..q_head_dim,
                ]);
                let k_chunk = k.clone().slice([
                    batch_index..batch_index + 1,
                    0..key_tokens,
                    0..heads,
                    0..k_head_dim,
                ]);
                let v_chunk = v.clone().slice([
                    batch_index..batch_index + 1,
                    0..key_tokens,
                    0..heads,
                    0..value_dim,
                ]);
                chunks.push(attention_once(
                    layout,
                    attention_mode,
                    &format!("{label}.batch_{batch_index:02}"),
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    head_dim,
                    chunk_tokens,
                    records,
                ));
            }
        }
        LayoutArg::Module => {
            let [_, heads, query_tokens, q_head_dim] = q.dims();
            let key_tokens = k.dims()[2];
            let k_head_dim = k.dims()[3];
            let value_dim = v.dims()[3];
            for batch_index in 0..batch {
                let q_chunk = q.clone().slice([
                    batch_index..batch_index + 1,
                    0..heads,
                    0..query_tokens,
                    0..q_head_dim,
                ]);
                let k_chunk = k.clone().slice([
                    batch_index..batch_index + 1,
                    0..heads,
                    0..key_tokens,
                    0..k_head_dim,
                ]);
                let v_chunk = v.clone().slice([
                    batch_index..batch_index + 1,
                    0..heads,
                    0..key_tokens,
                    0..value_dim,
                ]);
                chunks.push(attention_once(
                    layout,
                    attention_mode,
                    &format!("{label}.batch_{batch_index:02}"),
                    q_chunk,
                    k_chunk,
                    v_chunk,
                    head_dim,
                    chunk_tokens,
                    records,
                ));
            }
        }
    }

    Ok(Tensor::cat(chunks, 0))
}

fn attention_once<B: ForcedCubeAttentionBackend>(
    layout: LayoutArg,
    attention_mode: AttentionModeArg,
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    match layout {
        LayoutArg::Triposplat => match attention_mode {
            AttentionModeArg::Default => scaled_dot_product_attention_profiled(
                label,
                q,
                k,
                v,
                head_dim,
                chunk_tokens,
                records,
            ),
            AttentionModeArg::PrescaledDefault => {
                triposplat_full_prescaled_default_attention(label, q, k, v, head_dim, records)
            }
            AttentionModeArg::MaterializedDefault => {
                triposplat_full_materialized_default_attention(label, q, k, v, head_dim, records)
            }
            AttentionModeArg::ExplicitScale => {
                triposplat_full_explicit_scale_attention(label, q, k, v, head_dim, records)
            }
            AttentionModeArg::ExplicitScaleOne => {
                triposplat_full_scale_one_attention(label, q, k, v, head_dim, records)
            }
            AttentionModeArg::ExplicitScaleChunked => triposplat_chunked_explicit_scale_attention(
                label,
                q,
                k,
                v,
                head_dim,
                chunk_tokens,
                records,
            ),
            AttentionModeArg::DefaultModuleChunked => triposplat_chunked_default_module_attention(
                label,
                q,
                k,
                v,
                head_dim,
                chunk_tokens,
                records,
            ),
            AttentionModeArg::DefaultModuleContiguousChunked => {
                triposplat_chunked_default_module_contiguous_attention(
                    label,
                    q,
                    k,
                    v,
                    head_dim,
                    chunk_tokens,
                    records,
                )
            }
            AttentionModeArg::DefaultModuleFlattenedHeadsChunked => {
                triposplat_chunked_default_module_flattened_heads_attention(
                    label,
                    q,
                    k,
                    v,
                    head_dim,
                    chunk_tokens,
                    records,
                )
            }
            AttentionModeArg::DefaultModuleContiguousHybridChunked => {
                triposplat_chunked_default_module_contiguous_hybrid_attention(
                    label,
                    q,
                    k,
                    v,
                    head_dim,
                    chunk_tokens,
                    records,
                )
            }
            AttentionModeArg::DirectPrimitiveChunked => {
                triposplat_chunked_direct_attention(label, q, k, v, head_dim, chunk_tokens, records)
            }
            AttentionModeArg::DirectPrimitiveFlattenedHeadsChunked => {
                triposplat_chunked_direct_flattened_heads_attention(
                    label,
                    q,
                    k,
                    v,
                    head_dim,
                    chunk_tokens,
                    records,
                )
            }
            AttentionModeArg::CubeFallback
            | AttentionModeArg::CubeFlashUnit
            | AttentionModeArg::CubeFlashBlackbox2
            | AttentionModeArg::CubeFlashBlackbox4
            | AttentionModeArg::CubeFlashBlackbox8
            | AttentionModeArg::WgpuFlashRow
            | AttentionModeArg::WgpuFlashBlock => triposplat_chunked_forced_cube_attention(
                label,
                attention_mode,
                q,
                k,
                v,
                chunk_tokens,
                records,
            ),
        },
        LayoutArg::Module => {
            if matches!(
                attention_mode,
                AttentionModeArg::CubeFallback
                    | AttentionModeArg::CubeFlashUnit
                    | AttentionModeArg::CubeFlashBlackbox2
                    | AttentionModeArg::CubeFlashBlackbox4
                    | AttentionModeArg::CubeFlashBlackbox8
                    | AttentionModeArg::WgpuFlashRow
                    | AttentionModeArg::WgpuFlashBlock
            ) {
                return module_forced_cube_attention(label, attention_mode, q, k, v, records);
            }
            let q = match attention_mode {
                AttentionModeArg::PrescaledDefault => q.mul_scalar((head_dim as f64).powf(-0.5)),
                AttentionModeArg::MaterializedDefault => materialize_tensor(q),
                _ => q,
            };
            let (k, v) = if matches!(attention_mode, AttentionModeArg::MaterializedDefault) {
                (materialize_tensor(k), materialize_tensor(v))
            } else {
                (k, v)
            };
            module_attention(
                q,
                k,
                v,
                None,
                None,
                attention_options(attention_mode, head_dim),
            )
        }
    }
}

fn materialize_tensor<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone() + tensor.zeros_like()
}

fn force_contiguous_4d<B: Backend>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let dims = tensor.dims();
    tensor.flatten::<1>(0, 3).reshape(dims)
}

fn blackbox_query_pad_multiple(mode: AttentionModeArg) -> Option<usize> {
    match mode {
        AttentionModeArg::CubeFlashBlackbox2
        | AttentionModeArg::CubeFlashBlackbox4
        | AttentionModeArg::CubeFlashBlackbox8 => Some(128),
        _ => None,
    }
}

fn pad_triposplat_query_tokens<B: Backend>(
    q: Tensor<B, 4>,
    multiple: usize,
) -> (Tensor<B, 4>, usize) {
    let [batch, query_tokens, heads, head_dim] = q.dims();
    let padded_tokens = query_tokens.next_multiple_of(multiple);
    let pad_tokens = padded_tokens.saturating_sub(query_tokens);
    if pad_tokens == 0 {
        return (q, query_tokens);
    }

    let device = q.device();
    let dtype = q.dtype();
    assert!(
        dtype.is_float(),
        "CubeK blackbox query padding expects a float tensor, got {dtype:?}"
    );
    let pad = Tensor::<B, 4>::zeros([batch, pad_tokens, heads, head_dim], &device).cast(dtype);
    (Tensor::cat(vec![q, pad], 1), padded_tokens)
}

fn triposplat_full_forced_cube_attention<B: ForcedCubeAttentionBackend>(
    label: &str,
    mode: AttentionModeArg,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, head_dim] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let (q, padded_query_tokens) = match blackbox_query_pad_multiple(mode) {
        Some(multiple) => pad_triposplat_query_tokens(q, multiple),
        None => (q, query_tokens),
    };
    let q = force_contiguous_4d(q.permute([0, 2, 1, 3]));
    let k = force_contiguous_4d(k.permute([0, 2, 1, 3]));
    let v = force_contiguous_4d(v.permute([0, 2, 1, 3]));
    let out = B::forced_cube_attention_bhtd(mode, q, k, v)
        .permute([0, 2, 1, 3])
        .slice([0..batch, 0..query_tokens, 0..heads, 0..head_dim]);
    B::sync(&out.device()).expect("forced CubeCL TripoSplat attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(padded_query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some(format!("triposplat_forced_cubecl_attention_probe_{mode:?}")),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]->contiguous [B,H,T,D]; attention_mode={mode:?}; original_query_tokens={query_tokens}; padded_query_tokens={padded_query_tokens}",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_chunked_forced_cube_attention<B: ForcedCubeAttentionBackend>(
    label: &str,
    mode: AttentionModeArg,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_forced_cube_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            mode,
            q_chunk,
            k.clone(),
            v.clone(),
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn module_forced_cube_attention<B: ForcedCubeAttentionBackend>(
    label: &str,
    mode: AttentionModeArg,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, heads, query_tokens, head_dim] = q.dims();
    let key_tokens = k.dims()[2];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let out = B::forced_cube_attention_bhtd(mode, q, k, v);
    B::sync(&out.device()).expect("forced CubeCL module attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some(format!("module_forced_cubecl_attention_probe_{mode:?}")),
        backend_hint: Some(format!(
            "{}; layout=[B,H,T,D]; attention_mode={mode:?}",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_prescaled_default_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q
        .mul_scalar((head_dim as f64).powf(-0.5))
        .permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default())
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("prescaled-default attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_prescaled_default_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=PrescaledDefault",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_materialized_default_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = materialize_tensor(q.permute([0, 2, 1, 3]));
    let k = materialize_tensor(k.permute([0, 2, 1, 3]));
    let v = materialize_tensor(v.permute([0, 2, 1, 3]));
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default())
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("materialized-default attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_materialized_default_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=MaterializedDefault",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_chunked_explicit_scale_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_explicit_scale_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_default_module_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_default_module_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_default_module_contiguous_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_default_module_contiguous_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_default_module_flattened_heads_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_default_module_flattened_heads_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_default_module_contiguous_hybrid_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    const DEFAULT_MODULE_MIN_QUERY_TOKENS: usize = 512;

    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        let chunk_len = end.saturating_sub(start);
        if chunk_len >= DEFAULT_MODULE_MIN_QUERY_TOKENS {
            chunks.push(triposplat_full_default_module_contiguous_attention(
                &format!("{label}.chunk_{chunk_index:02}.default"),
                q_chunk,
                k.clone(),
                v.clone(),
                head_dim,
                records,
            ));
        } else {
            chunks.push(triposplat_full_direct_attention(
                &format!("{label}.chunk_{chunk_index:02}.direct_tail"),
                q_chunk,
                k.clone(),
                v.clone(),
                head_dim,
                records,
            ));
        }
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_direct_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_direct_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_chunked_direct_flattened_heads_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    chunk_tokens: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, actual_head_dim] = q.dims();
    let chunk_tokens = chunk_tokens.max(1);
    let mut chunks = Vec::with_capacity(query_tokens.div_ceil(chunk_tokens));
    for (chunk_index, start) in (0..query_tokens).step_by(chunk_tokens).enumerate() {
        let end = start.saturating_add(chunk_tokens).min(query_tokens);
        let q_chunk = q
            .clone()
            .slice([0..batch, start..end, 0..heads, 0..actual_head_dim]);
        chunks.push(triposplat_full_direct_flattened_heads_attention(
            &format!("{label}.chunk_{chunk_index:02}"),
            q_chunk,
            k.clone(),
            v.clone(),
            head_dim,
            records,
        ));
    }
    Tensor::cat(chunks, 1)
}

fn triposplat_full_default_module_contiguous_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = force_contiguous_4d(q.permute([0, 2, 1, 3]));
    let k = force_contiguous_4d(k.permute([0, 2, 1, 3]));
    let v = force_contiguous_4d(v.permute([0, 2, 1, 3]));
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default())
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("contiguous default-module attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_contiguous_default_module_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=DefaultModuleContiguousChunked",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_default_module_flattened_heads_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = force_contiguous_4d(q.permute([0, 2, 1, 3])).reshape([
        batch * heads,
        1,
        query_tokens,
        head_dim,
    ]);
    let k = force_contiguous_4d(k.permute([0, 2, 1, 3])).reshape([
        batch * heads,
        1,
        key_tokens,
        head_dim,
    ]);
    let v = force_contiguous_4d(v.permute([0, 2, 1, 3])).reshape([
        batch * heads,
        1,
        key_tokens,
        head_dim,
    ]);
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default())
        .reshape([batch, heads, query_tokens, head_dim])
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("flattened-heads default-module attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some(
            "triposplat_flattened_heads_default_module_attention_probe".to_string(),
        ),
        backend_hint: Some(format!(
            "{}; layout=[B*T,H,D] flattened to [B*H,1,T,D]; attention_mode=DefaultModuleFlattenedHeadsChunked",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_default_module_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let out = module_attention(q, k, v, None, None, AttentionModuleOptions::default())
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("default-module attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_default_module_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=DefaultModuleChunked",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_explicit_scale_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let out = module_attention(
        q,
        k,
        v,
        None,
        None,
        attention_options(AttentionModeArg::ExplicitScale, head_dim),
    )
    .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("explicit-scale attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_full_explicit_scale_fallback_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=ExplicitScale",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_scale_one_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let out = module_attention(
        q,
        k,
        v,
        None,
        None,
        attention_options(AttentionModeArg::ExplicitScaleOne, head_dim),
    )
    .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("scale-one attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_full_scale_one_fallback_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=ExplicitScaleOne",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_direct_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let score_elems = batch
        .saturating_mul(heads)
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q.permute([0, 2, 1, 3]);
    let k = k.permute([0, 2, 1, 3]);
    let v = v.permute([0, 2, 1, 3]);
    let scores = q
        .matmul(k.swap_dims(2, 3))
        .mul_scalar((head_dim as f64).powf(-0.5));
    let attn = softmax(scores, 3);
    let out = attn.matmul(v).permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("direct primitive attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_direct_primitive_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=DirectPrimitiveChunked",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn triposplat_full_direct_flattened_heads_attention<B: Backend>(
    label: &str,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    head_dim: usize,
    records: &mut Vec<TripoSplatProfileRecord>,
) -> Tensor<B, 4> {
    let [batch, query_tokens, heads, _] = q.dims();
    let key_tokens = k.dims()[1];
    let flat_batch = batch.saturating_mul(heads);
    let score_elems = flat_batch
        .saturating_mul(query_tokens)
        .saturating_mul(key_tokens);
    let start = Instant::now();
    let q = q
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, query_tokens, head_dim]);
    let k = k
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, key_tokens, head_dim]);
    let v = v
        .permute([0, 2, 1, 3])
        .reshape([flat_batch, key_tokens, head_dim]);
    let scores = q
        .matmul(k.swap_dims(1, 2))
        .mul_scalar((head_dim as f64).powf(-0.5));
    let attn = softmax(scores, 2);
    let out = attn
        .matmul(v)
        .reshape([batch, heads, query_tokens, head_dim])
        .permute([0, 2, 1, 3]);
    B::sync(&out.device()).expect("direct flattened-head attention profile sync failed");
    records.push(TripoSplatProfileRecord {
        label: label.to_string(),
        batch,
        tokens: query_tokens,
        channels: heads.saturating_mul(head_dim),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        key_tokens: Some(key_tokens),
        heads: Some(heads),
        head_dim: Some(head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(query_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: Some(format!("{:?}", out.dtype())),
        attention_path: Some("triposplat_direct_flattened_heads_attention_probe".to_string()),
        backend_hint: Some(format!(
            "{}; layout=[B,T,H,D]; attention_mode=DirectPrimitiveFlattenedHeadsChunked",
            B::name(&out.device())
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
    out
}

fn attention_options(mode: AttentionModeArg, head_dim: usize) -> AttentionModuleOptions {
    match mode {
        AttentionModeArg::Default
        | AttentionModeArg::PrescaledDefault
        | AttentionModeArg::MaterializedDefault
        | AttentionModeArg::DefaultModuleChunked
        | AttentionModeArg::DefaultModuleContiguousChunked
        | AttentionModeArg::DefaultModuleFlattenedHeadsChunked
        | AttentionModeArg::DefaultModuleContiguousHybridChunked
        | AttentionModeArg::CubeFallback
        | AttentionModeArg::CubeFlashUnit
        | AttentionModeArg::CubeFlashBlackbox2
        | AttentionModeArg::CubeFlashBlackbox4
        | AttentionModeArg::CubeFlashBlackbox8
        | AttentionModeArg::WgpuFlashRow
        | AttentionModeArg::WgpuFlashBlock => AttentionModuleOptions::default(),
        AttentionModeArg::ExplicitScaleOne => AttentionModuleOptions {
            scale: Some(1.0),
            ..Default::default()
        },
        AttentionModeArg::ExplicitScale
        | AttentionModeArg::ExplicitScaleChunked
        | AttentionModeArg::DirectPrimitiveChunked
        | AttentionModeArg::DirectPrimitiveFlattenedHeadsChunked => AttentionModuleOptions {
            scale: Some((head_dim as f64).powf(-0.5)),
            ..Default::default()
        },
    }
}

fn compare_attention_outputs<B: Backend>(
    reference: Tensor<B, 4>,
    candidate: Tensor<B, 4>,
) -> Result<AttentionDiffSummary, Box<dyn std::error::Error>> {
    let reference_dims = reference.dims();
    let candidate_dims = candidate.dims();
    if reference_dims != candidate_dims {
        return Err(format!(
            "attention compare shape mismatch: reference={reference_dims:?} candidate={candidate_dims:?}"
        )
        .into());
    }
    let reference_values = reference
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read reference attention output: {err:?}"))?;
    let candidate_values = candidate
        .to_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read candidate attention output: {err:?}"))?;
    if reference_values.len() != candidate_values.len() {
        return Err(format!(
            "attention compare length mismatch: reference={} candidate={}",
            reference_values.len(),
            candidate_values.len()
        )
        .into());
    }
    let mut max_abs = 0.0f64;
    let mut max_abs_flat_index = None;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut count_abs_gt_0_001 = 0usize;
    let mut count_abs_gt_0_01 = 0usize;
    let mut count_abs_gt_0_1 = 0usize;
    for (index, (&reference, &candidate)) in reference_values
        .iter()
        .zip(candidate_values.iter())
        .enumerate()
    {
        let diff = candidate as f64 - reference as f64;
        let abs = diff.abs();
        if abs > max_abs {
            max_abs = abs;
            max_abs_flat_index = Some(index);
        }
        sum_abs += abs;
        sum_sq += diff * diff;
        if abs > 1.0e-3 {
            count_abs_gt_0_001 += 1;
        }
        if abs > 1.0e-2 {
            count_abs_gt_0_01 += 1;
        }
        if abs > 1.0e-1 {
            count_abs_gt_0_1 += 1;
        }
    }
    let count = reference_values.len().max(1) as f64;
    Ok(AttentionDiffSummary {
        max_abs,
        mean_abs: sum_abs / count,
        rms: (sum_sq / count).sqrt(),
        count_abs_gt_0_001,
        count_abs_gt_0_01,
        count_abs_gt_0_1,
        max_abs_flat_index,
        reference_at_max_abs: max_abs_flat_index.map(|index| reference_values[index]),
        candidate_at_max_abs: max_abs_flat_index.map(|index| candidate_values[index]),
    })
}

fn push_direct_module_profile_record<B: Backend>(
    records: &mut Vec<TripoSplatProfileRecord>,
    label: String,
    dims: AttentionTensorDims,
    attention_mode: AttentionModeArg,
    chunk_tokens: usize,
    elapsed_ms: f64,
    device: &B::Device,
) {
    let score_elems = dims
        .batch
        .saturating_mul(dims.heads)
        .saturating_mul(dims.query_tokens)
        .saturating_mul(dims.key_tokens);
    records.push(TripoSplatProfileRecord {
        label,
        batch: dims.batch,
        tokens: dims.query_tokens,
        channels: dims.heads.saturating_mul(dims.head_dim),
        elapsed_ms,
        key_tokens: Some(dims.key_tokens),
        heads: Some(dims.heads),
        head_dim: Some(dims.head_dim),
        score_elems: Some(score_elems),
        query_chunk_tokens: Some(chunk_tokens),
        query_chunks: Some(1),
        dense_calls: Some(1),
        dtype: None,
        attention_path: Some(format!(
            "direct_public_module_attention_bhtd_{:?}",
            attention_mode
        )),
        backend_hint: Some(format!(
            "{}; layout=[B,H,T,D]; attention_mode={:?}",
            B::name(device),
            attention_mode
        )),
        finite_checked: None,
        nonfinite_count: None,
        first_nonfinite_index: None,
        min_finite: None,
        max_finite: None,
        mean_finite: None,
        rms_finite: None,
        max_abs_finite: None,
        finite_error: None,
    });
}

fn parse_query_chunk_tokens(raw: &str, query_tokens: usize) -> Result<(String, usize), String> {
    let value = raw.trim();
    if value.eq_ignore_ascii_case("full") {
        return Ok(("full".to_string(), query_tokens.max(1)));
    }
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("invalid --query-chunk-tokens value '{value}': {err}"))?;
    if parsed == 0 {
        return Err("--query-chunk-tokens values must be > 0".to_string());
    }
    Ok((parsed.to_string(), parsed))
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn percentile(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() as f64 - 1.0) * q).round() as usize;
    values[index.min(values.len() - 1)]
}

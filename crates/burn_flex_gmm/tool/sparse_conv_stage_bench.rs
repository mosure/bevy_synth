use std::time::Instant;

use burn::tensor::Tensor;
use burn_flex_gmm::SparseSubmConvConfig;
use burn_flex_gmm::wgpu::{
    DefaultWgpuBackend, NeighborDeviceAlgoPreference, SparseWgpuForwardConfig,
    SparseWgpuKernelVariant, clear_neighbor_rows_tensor_cache,
    neighbor_rows_tensor_from_coords_with_algo, reset_sparse_wgpu_kernel_stats,
    resolve_sparse_wgpu_forward_config, sparse_subm_conv_forward_wgpu_im2col_matmul,
    sparse_subm_conv_forward_wgpu_with_config, sparse_wgpu_kernel_stats,
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

fn parse_usize_arg(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    let flag = format!("--{name}");
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == flag {
            return args[i + 1]
                .parse::<usize>()
                .map_err(|err| format!("invalid value for {flag}: {err}"));
        }
        i += 1;
    }
    Ok(default)
}

fn parse_opt_usize_arg(args: &[String], name: &str) -> Result<Option<usize>, String> {
    let flag = format!("--{name}");
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == flag {
            let raw = &args[i + 1];
            if raw == "auto" {
                return Ok(None);
            }
            let value = raw
                .parse::<usize>()
                .map_err(|err| format!("invalid value for {flag}: {err}"))?;
            return Ok(Some(value));
        }
        i += 1;
    }
    Ok(None)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BenchVariant {
    Sparse(SparseWgpuKernelVariant),
    Im2ColMatmul,
}

fn parse_variant(args: &[String]) -> Result<BenchVariant, String> {
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == "--variant" {
            return match args[i + 1].as_str() {
                "auto" => Ok(BenchVariant::Sparse(SparseWgpuKernelVariant::Auto)),
                "baseline" => Ok(BenchVariant::Sparse(SparseWgpuKernelVariant::Baseline)),
                "fused" | "fused-oc4" => {
                    Ok(BenchVariant::Sparse(SparseWgpuKernelVariant::FusedOc4))
                }
                "im2col" | "im2col-matmul" => Ok(BenchVariant::Im2ColMatmul),
                other => Err(format!(
                    "unsupported --variant='{other}', expected auto|baseline|fused|im2col"
                )),
            };
        }
        i += 1;
    }
    Ok(BenchVariant::Sparse(SparseWgpuKernelVariant::Auto))
}

fn parse_neighbor_algo(args: &[String]) -> Result<NeighborDeviceAlgoPreference, String> {
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == "--neighbor" {
            return match args[i + 1].as_str() {
                "auto" => Ok(NeighborDeviceAlgoPreference::Auto),
                "scan" => Ok(NeighborDeviceAlgoPreference::Scan),
                "sorted" | "sorted-hash" => Ok(NeighborDeviceAlgoPreference::SortedHash),
                "hash" | "hash-table" => Ok(NeighborDeviceAlgoPreference::HashTableSerial),
                "bucket" | "bucket-hash" => Ok(NeighborDeviceAlgoPreference::BucketHash),
                other => Err(format!(
                    "unsupported --neighbor='{other}', expected auto|scan|sorted|hash|bucket"
                )),
            };
        }
        i += 1;
    }
    Ok(NeighborDeviceAlgoPreference::Auto)
}

fn line_coords(count: usize) -> Vec<[u32; 4]> {
    (0..count as u32).map(|x| [0, x, 0, 0]).collect()
}

fn config(
    rows: usize,
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
) -> SparseSubmConvConfig {
    let _ = rows;
    SparseSubmConvConfig {
        in_channels,
        out_channels,
        kernel_d: kernel,
        kernel_h: kernel,
        kernel_w: kernel,
        in_channels_per_group: in_channels,
        out_channels_per_group: out_channels,
        groups: 1,
        axis_order: [0, 1, 2],
        axis_sign: [1, 1, 1],
    }
}

fn percentile_ms(mut values: Vec<f64>, q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() as f64 - 1.0) * q).round() as usize;
    values[idx.min(values.len() - 1)]
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let rows = parse_usize_arg(args.as_slice(), "rows", 4096)?;
    let kernel = parse_usize_arg(args.as_slice(), "kernel", 3)?;
    let in_channels = parse_usize_arg(args.as_slice(), "in-ch", 64)?;
    let out_channels = parse_usize_arg(args.as_slice(), "out-ch", 128)?;
    let warmup = parse_usize_arg(args.as_slice(), "warmup", 1)?;
    let iters = parse_usize_arg(args.as_slice(), "iters", 3)?;
    let split_k = parse_opt_usize_arg(args.as_slice(), "split-k")?;
    let variant = parse_variant(args.as_slice())?;
    let neighbor_algo = parse_neighbor_algo(args.as_slice())?;

    if rows == 0 {
        return Err("--rows must be > 0".to_string());
    }
    if kernel == 0 || kernel % 2 == 0 {
        return Err("--kernel must be odd and > 0".to_string());
    }
    if in_channels == 0 || out_channels == 0 {
        return Err("--in-ch and --out-ch must be > 0".to_string());
    }
    if iters == 0 {
        return Err("--iters must be > 0".to_string());
    }

    let cfg = config(rows, kernel, in_channels, out_channels);
    let coords = line_coords(rows);
    let mut rng = Lcg::new(12345);
    let input: Vec<f32> = (0..rows.saturating_mul(in_channels))
        .map(|_| rng.next_f32())
        .collect();
    let weight_len = out_channels
        .saturating_mul(kernel)
        .saturating_mul(kernel)
        .saturating_mul(kernel)
        .saturating_mul(in_channels);
    let weight: Vec<f32> = (0..weight_len).map(|_| rng.next_f32()).collect();
    let bias: Vec<f32> = (0..out_channels).map(|_| rng.next_f32()).collect();

    let device = burn_wgpu::WgpuDevice::default();
    let input_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(input.as_slice(), &device)
        .reshape([rows, in_channels]);
    let weight_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(weight.as_slice(), &device)
        .reshape([out_channels, kernel, kernel, kernel, in_channels]);
    let bias_t = Tensor::<DefaultWgpuBackend, 1>::from_floats(bias.as_slice(), &device);

    clear_neighbor_rows_tensor_cache();
    let neighbor_t = neighbor_rows_tensor_from_coords_with_algo(
        &cfg,
        coords.as_slice(),
        &device,
        neighbor_algo,
    )?;

    let sparse_variant = match variant {
        BenchVariant::Sparse(variant) => variant,
        BenchVariant::Im2ColMatmul => SparseWgpuKernelVariant::Auto,
    };
    let forward = SparseWgpuForwardConfig {
        kernel_variant: sparse_variant,
        split_k,
    };
    let resolved = resolve_sparse_wgpu_forward_config(&cfg, rows, forward)?;

    let mut run_ms = Vec::with_capacity(iters);
    reset_sparse_wgpu_kernel_stats();
    for step in 0..(warmup + iters) {
        let start = Instant::now();
        let out = match variant {
            BenchVariant::Sparse(_) => sparse_subm_conv_forward_wgpu_with_config(
                &cfg,
                input_t.clone(),
                neighbor_t.clone(),
                weight_t.clone(),
                bias_t.clone(),
                forward,
            )?,
            BenchVariant::Im2ColMatmul => sparse_subm_conv_forward_wgpu_im2col_matmul(
                &cfg,
                input_t.clone(),
                neighbor_t.clone(),
                weight_t.clone(),
                bias_t.clone(),
            )?,
        };
        let _ = out.to_data();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        if step >= warmup {
            run_ms.push(elapsed_ms);
        }
    }
    let stats = sparse_wgpu_kernel_stats();

    let mean_ms = if run_ms.is_empty() {
        0.0
    } else {
        run_ms.iter().sum::<f64>() / run_ms.len() as f64
    };
    let min_ms = run_ms.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let p50_ms = percentile_ms(run_ms.clone(), 0.50);
    let p90_ms = percentile_ms(run_ms.clone(), 0.90);

    let variant_name = match variant {
        BenchVariant::Sparse(SparseWgpuKernelVariant::Auto) => "auto",
        BenchVariant::Sparse(SparseWgpuKernelVariant::Baseline) => "baseline",
        BenchVariant::Sparse(SparseWgpuKernelVariant::FusedOc4) => "fused-oc4",
        BenchVariant::Im2ColMatmul => "im2col-matmul",
    };
    let split_name = split_k
        .map(|v| v.to_string())
        .unwrap_or_else(|| "auto".to_string());
    let neighbor_name = match neighbor_algo {
        NeighborDeviceAlgoPreference::Auto => "auto",
        NeighborDeviceAlgoPreference::Scan => "scan",
        NeighborDeviceAlgoPreference::SortedHash => "sorted-hash",
        NeighborDeviceAlgoPreference::HashTableSerial => "hash-table-serial",
        NeighborDeviceAlgoPreference::BucketHash => "bucket-hash",
    };

    println!(
        concat!(
            "{{\n",
            "  \"rows\": {},\n",
            "  \"kernel\": {},\n",
            "  \"in_channels\": {},\n",
            "  \"out_channels\": {},\n",
            "  \"warmup\": {},\n",
            "  \"iters\": {},\n",
            "  \"variant\": \"{}\",\n",
            "  \"split_k\": \"{}\",\n",
            "  \"resolved_variant\": \"{}\",\n",
            "  \"resolved_split_k\": {},\n",
            "  \"neighbor\": \"{}\",\n",
            "  \"mean_ms\": {:.3},\n",
            "  \"min_ms\": {:.3},\n",
            "  \"p50_ms\": {:.3},\n",
            "  \"p90_ms\": {:.3},\n",
            "  \"dispatches_total\": {},\n",
            "  \"splitk_calls\": {},\n",
            "  \"fused_calls\": {},\n",
            "  \"single_group_specialized_calls\": {},\n",
            "  \"rows_total\": {},\n",
            "  \"output_elements_total\": {}\n",
            "}}"
        ),
        rows,
        kernel,
        in_channels,
        out_channels,
        warmup,
        iters,
        variant_name,
        split_name,
        match resolved.kernel_variant {
            SparseWgpuKernelVariant::Auto => "auto",
            SparseWgpuKernelVariant::Baseline => "baseline",
            SparseWgpuKernelVariant::FusedOc4 => "fused-oc4",
        },
        resolved.split_k,
        neighbor_name,
        mean_ms,
        min_ms,
        p50_ms,
        p90_ms,
        stats.total_dispatches,
        stats.splitk_calls,
        stats.fused_variant_calls,
        stats.single_group_specialized_calls,
        stats.total_rows,
        stats.total_output_elements
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("sparse_conv_stage_bench failed: {err}");
        std::process::exit(1);
    }
}

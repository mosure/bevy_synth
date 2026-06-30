use std::time::Instant;

use burn_flex_gmm::SparseSubmConvConfig;
use burn_flex_gmm::wgpu::{
    NeighborDeviceAlgoPreference, clear_neighbor_rows_tensor_cache, neighbor_rows_build_stats,
    neighbor_rows_tensor_from_coords_with_algo, reset_neighbor_rows_build_stats,
};

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

fn parse_algo(args: &[String]) -> Result<NeighborDeviceAlgoPreference, String> {
    let mut i = 0usize;
    while i + 1 < args.len() {
        if args[i] == "--algo" {
            return match args[i + 1].as_str() {
                "auto" => Ok(NeighborDeviceAlgoPreference::Auto),
                "scan" => Ok(NeighborDeviceAlgoPreference::Scan),
                "sorted" | "sorted-hash" => Ok(NeighborDeviceAlgoPreference::SortedHash),
                "hash" | "hash-table" => Ok(NeighborDeviceAlgoPreference::HashTableSerial),
                "bucket" | "bucket-hash" => Ok(NeighborDeviceAlgoPreference::BucketHash),
                other => Err(format!(
                    "unsupported --algo='{other}', expected one of: auto|scan|sorted|hash|bucket"
                )),
            };
        }
        i += 1;
    }
    Ok(NeighborDeviceAlgoPreference::Auto)
}

fn grid_coords(count: usize) -> Vec<[u32; 4]> {
    if count == 0 {
        return Vec::new();
    }
    let side = (count as f64).cbrt().ceil() as u32;
    let mut coords = Vec::with_capacity(count);
    'outer: for z in 0..side {
        for y in 0..side {
            for x in 0..side {
                coords.push([0, x, y, z]);
                if coords.len() == count {
                    break 'outer;
                }
            }
        }
    }
    coords
}

fn config(kernel: usize, channels: usize) -> SparseSubmConvConfig {
    SparseSubmConvConfig {
        in_channels: channels,
        out_channels: channels,
        kernel_d: kernel,
        kernel_h: kernel,
        kernel_w: kernel,
        in_channels_per_group: channels,
        out_channels_per_group: channels,
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
    let channels = parse_usize_arg(args.as_slice(), "channels", 8)?;
    let warmup = parse_usize_arg(args.as_slice(), "warmup", 1)?;
    let iters = parse_usize_arg(args.as_slice(), "iters", 3)?;
    let algo = parse_algo(args.as_slice())?;

    if rows == 0 {
        return Err("--rows must be > 0".to_string());
    }
    if kernel == 0 || kernel % 2 == 0 {
        return Err("--kernel must be odd and > 0".to_string());
    }
    if channels == 0 {
        return Err("--channels must be > 0".to_string());
    }
    if iters == 0 {
        return Err("--iters must be > 0".to_string());
    }

    let cfg = config(kernel, channels);
    let coords = grid_coords(rows);
    let device = burn_wgpu::WgpuDevice::default();

    let mut run_ms = Vec::with_capacity(iters);
    let mut probe_totals = Vec::with_capacity(iters);
    let mut probe_maxes = Vec::with_capacity(iters);

    for step in 0..(warmup + iters) {
        clear_neighbor_rows_tensor_cache();
        reset_neighbor_rows_build_stats();

        let start = Instant::now();
        let neighbor =
            neighbor_rows_tensor_from_coords_with_algo(&cfg, coords.as_slice(), &device, algo)?;
        let _ = neighbor.to_data();
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        let stats = neighbor_rows_build_stats();

        if step >= warmup {
            run_ms.push(elapsed_ms);
            probe_totals.push(stats.device_hash_probe_total);
            probe_maxes.push(stats.device_hash_probe_max);
        }
    }

    let mean_ms = if run_ms.is_empty() {
        0.0
    } else {
        run_ms.iter().sum::<f64>() / run_ms.len() as f64
    };
    let min_ms = run_ms.iter().copied().reduce(f64::min).unwrap_or(0.0);
    let p50_ms = percentile_ms(run_ms.clone(), 0.50);
    let p90_ms = percentile_ms(run_ms.clone(), 0.90);
    let probe_total_mean = if probe_totals.is_empty() {
        0.0
    } else {
        probe_totals.iter().sum::<u64>() as f64 / probe_totals.len() as f64
    };
    let probe_max_max = probe_maxes.into_iter().max().unwrap_or(0);

    let algo_name = match algo {
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
            "  \"channels\": {},\n",
            "  \"warmup\": {},\n",
            "  \"iters\": {},\n",
            "  \"algo\": \"{}\",\n",
            "  \"mean_ms\": {:.3},\n",
            "  \"min_ms\": {:.3},\n",
            "  \"p50_ms\": {:.3},\n",
            "  \"p90_ms\": {:.3},\n",
            "  \"hash_probe_total_mean\": {:.3},\n",
            "  \"hash_probe_max_max\": {}\n",
            "}}"
        ),
        rows,
        kernel,
        channels,
        warmup,
        iters,
        algo_name,
        mean_ms,
        min_ms,
        p50_ms,
        p90_ms,
        probe_total_mean,
        probe_max_max
    );

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("neighbor_stage_bench failed: {err}");
        std::process::exit(1);
    }
}

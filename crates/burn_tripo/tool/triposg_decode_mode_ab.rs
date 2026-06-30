#![recursion_limit = "256"]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::prelude::*;
use burn::tensor::{Int, TensorData};
use burn_foreground::pipeline::{PrepareImageConfig, RmbgPipeline, prepare_image_tensor};
use burn_tripo::pipeline::mesh::{DenseGrid, Mesh, grid_to_mesh};
use burn_tripo::pipeline::triposg::TripoSGPipeline;
use serde_json::json;

type BackendImpl = burn_wgpu::Wgpu;

fn main() {
    let handle = std::thread::Builder::new()
        .stack_size(512 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn worker thread");
    handle.join().expect("worker thread panicked");
}

fn run() {
    let device = <BackendImpl as burn::tensor::backend::BackendTypes>::Device::default();

    let weights_root = resolve_weights_root(
        "TRIPOSG_WEIGHTS_ROOT",
        "./crates/burn_tripo/assets/models/MIDI-3D",
    )
    .expect("TripoSG weights not found");
    let rmbg_root = resolve_weights_root(
        "RMBG_WEIGHTS_ROOT",
        "./crates/burn_foreground/assets/models/RMBG-1.4",
    )
    .expect("RMBG-1.4 weights not found");
    let image_path = PathBuf::from(
        std::env::var("TRIPOSG_AB_IMAGE").unwrap_or_else(|_| "docs/input_chair.jpg".to_string()),
    );
    let out_device = PathBuf::from(
        std::env::var("TRIPOSG_AB_OUT_DEVICE")
            .unwrap_or_else(|_| "docs/chair_triposg_decode_device_scatter.glb".to_string()),
    );
    let out_host = PathBuf::from(
        std::env::var("TRIPOSG_AB_OUT_HOST")
            .unwrap_or_else(|_| "docs/chair_triposg_decode_host_chunked.glb".to_string()),
    );

    let steps = env_usize("TRIPOSG_AB_STEPS", 2);
    let tokens = env_usize("TRIPOSG_AB_TOKENS", 128);
    let guidance = env_f32("TRIPOSG_AB_GUIDANCE", 7.0);
    let resolution = env_usize("TRIPOSG_AB_RES", 64);
    let chunk_size = env_usize("TRIPOSG_AB_CHUNK", 4096);
    let bounds = [-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0];
    let decode_warmup = env_usize("TRIPOSG_AB_DECODE_WARMUP", 1);
    let decode_iters = env_usize("TRIPOSG_AB_DECODE_ITERS", 5).max(1);
    let chunk_sweep = parse_chunk_sweep();

    println!("triposg_decode_mode_ab");
    println!("- backend: wgpu");
    println!("- image: {}", image_path.display());
    println!("- weights: {}", weights_root.display());
    println!("- rmbg: {}", rmbg_root.display());
    println!("- steps={steps} tokens={tokens} guidance={guidance}");
    println!("- resolution={resolution} chunk_size={chunk_size}");
    println!(
        "- decode_bench warmup_iters={} measure_iters={} chunk_sweep={:?}",
        decode_warmup, decode_iters, chunk_sweep
    );

    let mut pipeline = TripoSGPipeline::<BackendImpl>::from_pretrained(weights_root, &device)
        .expect("failed to load TripoSG");
    let rmbg = RmbgPipeline::<BackendImpl>::from_pretrained(rmbg_root, &device)
        .expect("failed to load RMBG-1.4");

    let image_tensor = prepare_image_tensor::<BackendImpl>(
        image_path.as_path(),
        Some(&rmbg),
        &device,
        &PrepareImageConfig::default(),
    )
    .expect("failed to prepare input image");
    let processed = pipeline.image_processor.preprocess(image_tensor);
    let embeds = pipeline
        .image_encoder
        .as_ref()
        .expect("image encoder unavailable")
        .forward(processed);

    let sample_start = Instant::now();
    let output = pipeline.sample_from_embeds(embeds, 1, steps, tokens, guidance, None, None);
    <BackendImpl as Backend>::sync(&device);
    let sample_ms = elapsed_ms(sample_start);
    println!("- diffusion/sample_ms={sample_ms:.2}");

    let latents = output.latents;

    let device_values = decode_grid_values_device_scatter(
        &latents,
        &pipeline.vae,
        bounds,
        resolution,
        chunk_size,
        &device,
    );
    <BackendImpl as Backend>::sync(&device);
    let host_values = decode_grid_values_host_chunked(
        &latents,
        &pipeline.vae,
        bounds,
        resolution,
        chunk_size,
        &device,
    );
    <BackendImpl as Backend>::sync(&device);
    let cat_values = decode_grid_values_device_cat(
        &latents,
        &pipeline.vae,
        bounds,
        resolution,
        chunk_size,
        &device,
    );
    <BackendImpl as Backend>::sync(&device);

    let (mean_abs, max_abs, mse) = compare_stats(device_values.as_slice(), host_values.as_slice());
    let (cat_mean_abs, cat_max_abs, cat_mse) =
        compare_stats(cat_values.as_slice(), host_values.as_slice());
    println!("- mode_delta mean_abs={mean_abs:.6e} max_abs={max_abs:.6e} mse={mse:.6e}");
    println!(
        "- mode_delta_device_cat_vs_host mean_abs={cat_mean_abs:.6e} max_abs={cat_max_abs:.6e} mse={cat_mse:.6e}"
    );

    benchmark_decode_modes(
        &latents,
        &pipeline.vae,
        bounds,
        resolution,
        chunk_size,
        decode_warmup,
        decode_iters,
        &chunk_sweep,
        &device,
    );

    let device_grid = DenseGrid {
        values: device_values,
        size: [resolution, resolution, resolution],
        bounds,
    };
    let host_grid = DenseGrid {
        values: host_values,
        size: [resolution, resolution, resolution],
        bounds,
    };

    let device_mesh = grid_to_mesh(&device_grid, 0.0).expect("device mesh extraction failed");
    let host_mesh = grid_to_mesh(&host_grid, 0.0).expect("host mesh extraction failed");
    println!(
        "- mesh_device vertices={} faces={}",
        device_mesh.vertices.len(),
        device_mesh.faces.len()
    );
    println!(
        "- mesh_host vertices={} faces={}",
        host_mesh.vertices.len(),
        host_mesh.faces.len()
    );

    write_minimal_glb(out_device.as_path(), &device_mesh).expect("failed to write device GLB");
    write_minimal_glb(out_host.as_path(), &host_mesh).expect("failed to write host GLB");
    println!("- wrote {}", out_device.display());
    println!("- wrote {}", out_host.display());
}

fn decode_grid_values_device_scatter<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    device: &B::Device,
) -> Vec<f32> {
    let total = resolution * resolution * resolution;
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunk_indices = Vec::<i32>::with_capacity(chunk_size);
    let mut values = Tensor::<B, 1>::zeros([total], device);

    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);
        chunk_indices.push(idx as i32);

        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        values = write_scatter_chunk(
            latents,
            vae,
            coords.as_slice(),
            chunk_indices.as_slice(),
            device,
            values,
        );
        coords.clear();
        chunk_indices.clear();
    }

    if !coords.is_empty() {
        values = write_scatter_chunk(
            latents,
            vae,
            coords.as_slice(),
            chunk_indices.as_slice(),
            device,
            values,
        );
    }

    values
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("failed to read device scatter values")
}

fn decode_grid_values_device_cat<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    device: &B::Device,
) -> Vec<f32> {
    let total = resolution * resolution * resolution;
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunks = Vec::<Tensor<B, 3>>::new();
    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);
        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        let coords_tensor = Tensor::<B, 1>::from_floats(coords.as_slice(), device)
            .reshape([count as i32, 3])
            .unsqueeze_dim(0);
        chunks.push(vae.decode(coords_tensor, latents, None));
        coords.clear();
    }

    if !coords.is_empty() {
        let count = coords.len() / 3;
        let coords_tensor = Tensor::<B, 1>::from_floats(coords.as_slice(), device)
            .reshape([count as i32, 3])
            .unsqueeze_dim(0);
        chunks.push(vae.decode(coords_tensor, latents, None));
    }

    if chunks.is_empty() {
        return Vec::new();
    }
    let decoded = if chunks.len() == 1 {
        chunks.pop().expect("single chunk exists")
    } else {
        Tensor::cat(chunks, 1)
    };
    decoded
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("failed to read device cat values")
}

fn decode_grid_values_host_chunked<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    chunk_size: usize,
    device: &B::Device,
) -> Vec<f32> {
    let total = resolution * resolution * resolution;
    let mut values = vec![0.0f32; total];
    let step_x = dense_grid_step(bounds[0], bounds[3], resolution);
    let step_y = dense_grid_step(bounds[1], bounds[4], resolution);
    let step_z = dense_grid_step(bounds[2], bounds[5], resolution);

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunk_start = 0usize;

    for idx in 0..total {
        let (x, y, z) = dense_grid_index_to_xyz(idx, resolution);
        coords.push(bounds[0] + step_x * x as f32);
        coords.push(bounds[1] + step_y * y as f32);
        coords.push(bounds[2] + step_z * z as f32);

        let count = coords.len() / 3;
        if count < chunk_size {
            continue;
        }
        let end = chunk_start + count;
        write_host_chunk(
            latents,
            vae,
            coords.as_slice(),
            device,
            &mut values[chunk_start..end],
        );
        coords.clear();
        chunk_start = end;
    }

    if !coords.is_empty() {
        let count = coords.len() / 3;
        let end = chunk_start + count;
        write_host_chunk(
            latents,
            vae,
            coords.as_slice(),
            device,
            &mut values[chunk_start..end],
        );
    }

    values
}

fn write_scatter_chunk<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    coords: &[f32],
    indices: &[i32],
    device: &B::Device,
    output: Tensor<B, 1>,
) -> Tensor<B, 1> {
    if coords.is_empty() {
        return output;
    }
    let count = coords.len() / 3;
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([count as i32, 3])
        .unsqueeze_dim(0);
    let decoded = vae
        .decode(coords_tensor, latents.clone(), None)
        .reshape([count]);
    let indices_tensor =
        Tensor::<B, 1, Int>::from_data(TensorData::new(indices.to_vec(), [count]), device);
    output.scatter(0, indices_tensor, decoded)
}

fn write_host_chunk<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    coords: &[f32],
    device: &B::Device,
    output_slice: &mut [f32],
) {
    if coords.is_empty() {
        return;
    }
    let count = coords.len() / 3;
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([count as i32, 3])
        .unsqueeze_dim(0);
    let decoded = vae.decode(coords_tensor, latents, None);
    let data = decoded
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("failed to read host chunk decode values");
    output_slice.copy_from_slice(&data[..output_slice.len()]);
}

fn dense_grid_step(start: f32, end: f32, steps: usize) -> f32 {
    if steps <= 1 {
        0.0
    } else {
        (end - start) / (steps as f32 - 1.0)
    }
}

fn dense_grid_index_to_xyz(index: usize, resolution: usize) -> (usize, usize, usize) {
    let plane = resolution * resolution;
    let z = index / plane;
    let rem = index - z * plane;
    let y = rem / resolution;
    let x = rem - y * resolution;
    (x, y, z)
}

fn compare_stats(lhs: &[f32], rhs: &[f32]) -> (f64, f64, f64) {
    let count = lhs.len().min(rhs.len());
    if count == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut max_abs = 0.0f64;
    for i in 0..count {
        let delta = (lhs[i] - rhs[i]) as f64;
        let abs = delta.abs();
        sum_abs += abs;
        sum_sq += delta * delta;
        max_abs = max_abs.max(abs);
    }
    let inv = 1.0 / count as f64;
    (sum_abs * inv, max_abs, sum_sq * inv)
}

#[allow(clippy::too_many_arguments)]
fn benchmark_decode_modes<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &burn_tripo::model::triposg::vae::TripoSGVae<B>,
    bounds: [f32; 6],
    resolution: usize,
    default_chunk_size: usize,
    warmup_iters: usize,
    measure_iters: usize,
    chunk_sweep: &[usize],
    device: &B::Device,
) {
    let total_points = resolution * resolution * resolution;
    let bytes_per_point = core::mem::size_of::<f32>();

    println!("- decode_bench_start");
    for chunk in chunk_sweep
        .iter()
        .copied()
        .chain(std::iter::once(default_chunk_size))
    {
        let chunk = chunk.max(1);
        let num_chunks = total_points.div_ceil(chunk);
        let host_readback_bytes = total_points.saturating_mul(bytes_per_point);
        let scatter_nominal_write_bytes = host_readback_bytes.saturating_mul(num_chunks);

        let scatter = bench_mode(
            || {
                let _ = decode_grid_values_device_scatter(
                    latents, vae, bounds, resolution, chunk, device,
                );
                B::sync(device);
            },
            warmup_iters,
            measure_iters,
        );
        let cat = bench_mode(
            || {
                let _ =
                    decode_grid_values_device_cat(latents, vae, bounds, resolution, chunk, device);
                B::sync(device);
            },
            warmup_iters,
            measure_iters,
        );
        let host = bench_mode(
            || {
                let _ = decode_grid_values_host_chunked(
                    latents, vae, bounds, resolution, chunk, device,
                );
                B::sync(device);
            },
            warmup_iters,
            measure_iters,
        );

        println!(
            "- decode_bench chunk={} chunks={} scatter_ms(mean/med/min/max)={:.2}/{:.2}/{:.2}/{:.2} cat_ms={:.2}/{:.2}/{:.2}/{:.2} host_ms={:.2}/{:.2}/{:.2}/{:.2} scatter_nominal_write_mib={:.2} host_readback_mib={:.2}",
            chunk,
            num_chunks,
            scatter.mean_ms,
            scatter.median_ms,
            scatter.min_ms,
            scatter.max_ms,
            cat.mean_ms,
            cat.median_ms,
            cat.min_ms,
            cat.max_ms,
            host.mean_ms,
            host.median_ms,
            host.min_ms,
            host.max_ms,
            scatter_nominal_write_bytes as f64 / (1024.0 * 1024.0),
            host_readback_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    println!("- decode_bench_end");
}

#[derive(Debug, Clone, Copy)]
struct BenchStats {
    mean_ms: f64,
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn bench_mode<F: FnMut()>(mut f: F, warmup_iters: usize, measure_iters: usize) -> BenchStats {
    for _ in 0..warmup_iters {
        f();
    }

    let mut samples = Vec::with_capacity(measure_iters);
    for _ in 0..measure_iters {
        let start = Instant::now();
        f();
        samples.push(elapsed_ms(start));
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));

    let min_ms = samples.first().copied().unwrap_or(0.0);
    let max_ms = samples.last().copied().unwrap_or(0.0);
    let mean_ms = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    let median_ms = if samples.is_empty() {
        0.0
    } else if samples.len() % 2 == 1 {
        samples[samples.len() / 2]
    } else {
        let hi = samples.len() / 2;
        (samples[hi - 1] + samples[hi]) * 0.5
    };

    BenchStats {
        mean_ms,
        median_ms,
        min_ms,
        max_ms,
    }
}

fn write_minimal_glb(path: &Path, mesh: &Mesh) -> Result<(), String> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err("cannot write empty mesh".to_string());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }

    let mut positions = Vec::with_capacity(mesh.vertices.len() * 12);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
        positions.extend_from_slice(&vertex[0].to_le_bytes());
        positions.extend_from_slice(&vertex[1].to_le_bytes());
        positions.extend_from_slice(&vertex[2].to_le_bytes());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 12);
    for face in &mesh.faces {
        indices.extend_from_slice(&face[0].to_le_bytes());
        indices.extend_from_slice(&face[1].to_le_bytes());
        indices.extend_from_slice(&face[2].to_le_bytes());
    }

    let mut bin = Vec::with_capacity(positions.len() + indices.len() + 8);
    let pos_offset = 0usize;
    bin.extend_from_slice(positions.as_slice());
    pad_4(&mut bin, 0);
    let pos_len = positions.len();
    let idx_offset = bin.len();
    bin.extend_from_slice(indices.as_slice());
    let idx_len = indices.len();
    pad_4(&mut bin, 0);

    let json_value = json!({
        "asset": {"version": "2.0", "generator": "triposg_decode_mode_ab"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": {"POSITION": 0},
                "indices": 1,
                "mode": 4
            }]
        }],
        "buffers": [{"byteLength": bin.len()}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": pos_offset, "byteLength": pos_len, "target": 34962},
            {"buffer": 0, "byteOffset": idx_offset, "byteLength": idx_len, "target": 34963}
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": mesh.vertices.len(),
                "type": "VEC3",
                "min": [min[0], min[1], min[2]],
                "max": [max[0], max[1], max[2]]
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": mesh.faces.len() * 3,
                "type": "SCALAR"
            }
        ]
    });
    let mut json_bytes =
        serde_json::to_vec(&json_value).map_err(|err| format!("json serialize failed: {err}"))?;
    pad_4(&mut json_bytes, 0x20);

    let json_chunk_len = json_bytes.len() as u32;
    let bin_chunk_len = bin.len() as u32;
    let total_len = 12u32 + 8 + json_chunk_len + 8 + bin_chunk_len;

    let mut glb = Vec::with_capacity(total_len as usize);
    glb.extend_from_slice(&0x46546C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total_len.to_le_bytes());

    glb.extend_from_slice(&json_chunk_len.to_le_bytes());
    glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes());
    glb.extend_from_slice(json_bytes.as_slice());

    glb.extend_from_slice(&bin_chunk_len.to_le_bytes());
    glb.extend_from_slice(&0x004E4942u32.to_le_bytes());
    glb.extend_from_slice(bin.as_slice());

    fs::write(path, glb).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

fn pad_4(buffer: &mut Vec<u8>, byte: u8) {
    let rem = buffer.len() % 4;
    if rem == 0 {
        return;
    }
    let pad = 4 - rem;
    buffer.extend(std::iter::repeat_n(byte, pad));
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn resolve_weights_root(env_var: &str, fallback: &str) -> Option<PathBuf> {
    if let Ok(value) = std::env::var(env_var) {
        let path = PathBuf::from(value);
        if let Some(root) = normalize_weights_root(path.as_path()) {
            return Some(root);
        }
    }
    normalize_weights_root(Path::new(fallback))
}

fn normalize_weights_root(path: &Path) -> Option<PathBuf> {
    if path.is_dir() {
        Some(path.to_path_buf())
    } else if path.is_file() {
        path.parent().map(Path::to_path_buf)
    } else {
        None
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(default)
}

fn parse_chunk_sweep() -> Vec<usize> {
    let raw = std::env::var("TRIPOSG_AB_CHUNK_SWEEP")
        .unwrap_or_else(|_| "512,1024,2048,4096,8192,16384".to_string());
    let mut out = raw
        .split(',')
        .filter_map(|token| token.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

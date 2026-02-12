use burn::prelude::*;
use burn::tensor::TensorData;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::model::triposg::vae::TripoSGVae;
use crate::pipeline::mesh::DenseGrid;
use crate::readback::{tensor_to_vec_f32, tensor_to_vec_i32};

const FLASH_INVALID_SENTINEL: f32 = -10000.0;
const FLASH_INVALID_THRESHOLD: f32 = -9000.0;
const FLASH_WGPU_MAX_POINTS: usize = 4096;
static FLASH_FORCE_CPU: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct HierarchicalExtractConfig {
    pub bounds: [f32; 6],
    pub dense_octree_depth: usize,
    pub hierarchical_octree_depth: usize,
    pub chunk_size: usize,
    pub band_threshold: f32,
}

impl HierarchicalExtractConfig {
    pub fn new(
        bounds: [f32; 6],
        dense_octree_depth: usize,
        hierarchical_octree_depth: usize,
    ) -> Self {
        Self {
            bounds,
            dense_octree_depth,
            hierarchical_octree_depth,
            chunk_size: 10_000,
            band_threshold: 1.0,
        }
    }
}

pub fn hierarchical_extract_geometry<B: Backend>(
    latents: Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &HierarchicalExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let dense_depth = config
        .dense_octree_depth
        .min(config.hierarchical_octree_depth);
    let chunk_size = config.chunk_size.max(1);
    let mut size = pow2(dense_depth);
    let bounds = config.bounds;
    let xs = linspace(bounds[0], bounds[3], size);
    let ys = linspace(bounds[1], bounds[4], size);
    let zs = linspace(bounds[2], bounds[5], size);

    let mut grid_values = eval_grid(&latents, vae, &xs, &ys, &zs, chunk_size)?;

    for depth in (dense_depth + 1)..=config.hierarchical_octree_depth {
        let next_size = pow2(depth);
        let mut high_values = upsample_nearest(&grid_values, size);

        let edge_coords = find_candidates_band(&grid_values, size, config.band_threshold);
        if !edge_coords.is_empty() {
            let expanded = expand_edge_region(&edge_coords, size, next_size);
            if !expanded.is_empty() {
                update_grid_from_coords(
                    &latents,
                    vae,
                    &expanded,
                    next_size,
                    bounds,
                    chunk_size,
                    &mut high_values,
                )?;
            }
        }

        grid_values = high_values;
        size = next_size;
    }

    Ok(DenseGrid {
        values: grid_values,
        size: [size, size, size],
        bounds,
    })
}

#[derive(Debug, Clone)]
pub struct FlashExtractConfig {
    pub bounds: [f32; 6],
    pub octree_depth: usize,
    pub num_chunks: usize,
    pub mc_level: f32,
    pub min_resolution: usize,
    pub mini_grid_num: usize,
}

impl FlashExtractConfig {
    pub fn new(bounds: [f32; 6], octree_depth: usize) -> Self {
        Self {
            bounds,
            octree_depth,
            num_chunks: 10_000,
            mc_level: 0.0,
            min_resolution: 63,
            mini_grid_num: 4,
        }
    }
}

pub fn flash_extract_geometry<B: Backend>(
    latents: Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    if !should_use_gpu_flash::<B>() {
        return flash_extract_geometry_cpu(latents, vae, config);
    }
    let latents_cpu = latents.clone();
    match flash_extract_geometry_gpu(latents, vae, config) {
        Ok(grid) => {
            if grid_is_all_nan(&grid) && std::env::var("TRIPOSG_FLASH_NO_FALLBACK").is_err() {
                eprintln!(
                    "flash_extract_geometry: GPU grid was all NaNs, retrying with CPU flash path."
                );
                FLASH_FORCE_CPU.store(true, Ordering::Relaxed);
                return flash_extract_geometry_cpu(latents_cpu, vae, config);
            }
            Ok(grid)
        }
        Err(err) => {
            if std::env::var("TRIPOSG_FLASH_NO_FALLBACK").is_err() {
                eprintln!(
                    "flash_extract_geometry: GPU flash failed ({err}), retrying with CPU flash path."
                );
                FLASH_FORCE_CPU.store(true, Ordering::Relaxed);
                return flash_extract_geometry_cpu(latents_cpu, vae, config);
            }
            Err(err)
        }
    }
}

fn should_use_gpu_flash<B: Backend>() -> bool {
    if std::env::var("TRIPOSG_FLASH_CPU").is_ok() {
        return false;
    }
    if FLASH_FORCE_CPU.load(Ordering::Relaxed) {
        return false;
    }
    if std::env::var("TRIPOSG_FLASH_GPU").is_ok() {
        return true;
    }
    let backend_name = std::any::type_name::<B>().to_ascii_lowercase();
    if backend_name.contains("ndarray") {
        return false;
    }
    true
}

fn flash_max_points<B: Backend>() -> usize {
    if let Ok(raw) = std::env::var("TRIPOSG_FLASH_MAX_POINTS")
        && let Ok(parsed) = raw.parse::<usize>()
    {
        return parsed.max(1);
    }
    let backend_name = std::any::type_name::<B>().to_ascii_lowercase();
    if backend_name.contains("wgpu") {
        return FLASH_WGPU_MAX_POINTS;
    }
    usize::MAX
}

fn flash_extract_geometry_gpu<B: Backend>(
    latents: Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let bounds = config.bounds;
    let octree_depth = config.octree_depth.max(1);
    let num_chunks = config.num_chunks.max(1);
    let min_resolution = config.min_resolution.max(2);
    let mini_grid_num = config.mini_grid_num.max(1);

    let resolutions = build_flash_resolutions(octree_depth, min_resolution, mini_grid_num);
    if resolutions.is_empty() {
        return Err("flash extractor produced empty resolution list".into());
    }

    let base_res = resolutions[0];
    let base_grid = base_res + 1;

    let latent_proj = vae.prepare_latent_projection(latents, None);
    let kv_cache = vae.build_kv_cache(latent_proj.clone(), None);

    let mut grid_logits = eval_flash_base_grid_gpu(
        vae,
        &latent_proj,
        &kv_cache,
        bounds,
        base_res,
        num_chunks,
        mini_grid_num,
    )?;
    let mut grid_size = base_grid;
    log_flash_stats("base", &grid_logits, grid_size);
    if grid_all_invalid(&grid_logits) {
        return Err("flash base grid decode returned only sentinel values".into());
    }

    let mut shared_kv_cache = Some(kv_cache);
    for (level_idx, &res) in resolutions.iter().enumerate().skip(1) {
        let next_size = res + 1;
        let step_x = (bounds[3] - bounds[0]) / res as f32;
        let step_y = (bounds[4] - bounds[1]) / res as f32;
        let step_z = (bounds[5] - bounds[2]) / res as f32;

        let device = grid_logits.device();
        let next_total = next_size * next_size * next_size;
        let mut next_logits = Tensor::<B, 1>::full([next_total], FLASH_INVALID_SENTINEL, &device);

        let mut curr_mask = extract_near_surface_mask_gpu(&grid_logits, config.mc_level);
        let near_mask = grid_logits.clone().abs().lower_elem(0.95);
        curr_mask = curr_mask.bool_or(near_mask);

        let expand_num = if level_idx == resolutions.len() - 1 {
            0
        } else {
            1
        };
        for _ in 0..expand_num {
            curr_mask = dilate_mask_gpu(curr_mask);
            curr_mask = dilate_mask_gpu(curr_mask);
        }

        let curr_coords = curr_mask.argwhere();
        let curr_count = curr_coords.shape().dims::<2>()[0];
        if curr_count == 0 {
            log_flash_level_empty("curr_mask", level_idx, grid_size, next_size);
            break;
        }

        let doubled = curr_coords.clone().mul_scalar(2);
        let doubled_indices = coords_to_linear_indices_2(doubled, next_size);
        let ones = Tensor::<B, 1>::ones([doubled_indices.shape().dims::<1>()[0]], &device);
        let mut next_index = Tensor::<B, 1>::zeros([next_total], &device);
        next_index = next_index.scatter(0, doubled_indices, ones);
        let mut next_index = next_index
            .reshape([next_size as i32, next_size as i32, next_size as i32])
            .greater_elem(0.0);

        for _ in 0..(2 - expand_num) {
            next_index = dilate_mask_gpu(next_index);
        }

        let next_coords = next_index.argwhere();
        let next_count = next_coords.shape().dims::<2>()[0];
        if next_count == 0 {
            log_flash_level_empty("next_mask", level_idx, grid_size, next_size);
            break;
        }

        let flat_indices = coords_to_linear_indices_2(next_coords.clone(), next_size);
        let world_coords = coords_to_world_2(next_coords, bounds, [step_x, step_y, step_z]);

        decode_flash_points_gpu(
            vae,
            &latent_proj,
            &mut shared_kv_cache,
            world_coords,
            flat_indices,
            num_chunks,
            &mut next_logits,
        )?;

        grid_logits = next_logits.reshape([next_size as i32, next_size as i32, next_size as i32]);
        grid_size = next_size;
        log_flash_stats(&format!("level-{level_idx}"), &grid_logits, grid_size);
    }

    let invalid = grid_logits
        .clone()
        .lower_equal_elem(FLASH_INVALID_THRESHOLD);
    let nan = Tensor::<B, 1>::from_floats([f32::NAN], &grid_logits.device()).reshape([1, 1, 1]);
    let grid_logits = grid_logits.mask_where(invalid, nan);

    let octree_resolution = 1usize << octree_depth;
    let sdf = grid_logits
        .mul_scalar(-1.0 / octree_resolution as f32)
        .permute([2, 1, 0]);
    let sdf_values =
        tensor_to_vec_f32(sdf).map_err(|err| format!("failed to read flash grid logits: {err}"))?;

    Ok(DenseGrid {
        values: sdf_values,
        size: [grid_size, grid_size, grid_size],
        bounds,
    })
}

fn grid_is_all_nan(grid: &DenseGrid) -> bool {
    grid.values.iter().all(|value| value.is_nan())
}

fn log_flash_stats<B: Backend>(label: &str, grid: &Tensor<B, 3>, size: usize) {
    if std::env::var("TRIPOSG_FLASH_DEBUG").is_err() {
        return;
    }
    let data = match tensor_to_vec_f32(grid.clone()) {
        Ok(values) => values,
        Err(_) => {
            eprintln!("flash_extract_geometry: failed to read grid logits for {label}");
            return;
        }
    };
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut nan_count = 0usize;
    for value in data {
        if value.is_nan() {
            nan_count += 1;
        } else {
            min = min.min(value);
            max = max.max(value);
        }
    }
    if min == f32::INFINITY {
        eprintln!("flash_extract_geometry[{label}]: grid {size}^3 all NaN");
    } else {
        eprintln!(
            "flash_extract_geometry[{label}]: grid {size}^3 min={min:.4} max={max:.4} nan={nan_count}"
        );
    }
}

fn grid_all_invalid<B: Backend>(grid: &Tensor<B, 3>) -> bool {
    let Ok(values) = tensor_to_vec_f32(grid.clone()) else {
        return false;
    };
    values
        .iter()
        .all(|value| value.is_nan() || *value <= FLASH_INVALID_THRESHOLD)
}

fn log_flash_level_empty(reason: &str, level_idx: usize, curr: usize, next: usize) {
    if std::env::var("TRIPOSG_FLASH_DEBUG").is_err() {
        return;
    }
    eprintln!(
        "flash_extract_geometry: empty {reason} at level {level_idx} (curr={curr}, next={next}), stopping refinement"
    );
}

fn flash_extract_geometry_cpu<B: Backend>(
    latents: Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let bounds = config.bounds;
    let octree_depth = config.octree_depth.max(1);
    let num_chunks = config.num_chunks.max(1);
    let min_resolution = config.min_resolution.max(2);
    let mini_grid_num = config.mini_grid_num.max(1);

    let resolutions = build_flash_resolutions(octree_depth, min_resolution, mini_grid_num);
    if resolutions.is_empty() {
        return Err("flash extractor produced empty resolution list".into());
    }

    let base_res = resolutions[0];
    let base_grid = base_res + 1;
    let (xs, ys, zs) = (
        linspace(bounds[0], bounds[3], base_grid),
        linspace(bounds[1], bounds[4], base_grid),
        linspace(bounds[2], bounds[5], base_grid),
    );

    let base_logits = eval_flash_base_grid(
        latents.clone(),
        vae,
        &xs,
        &ys,
        &zs,
        num_chunks,
        mini_grid_num,
    )?;

    let mut grid_logits = base_logits;
    let mut grid_size = base_grid;

    for (level_idx, &res) in resolutions.iter().enumerate().skip(1) {
        let next_size = res + 1;
        let step_x = (bounds[3] - bounds[0]) / res as f32;
        let step_y = (bounds[4] - bounds[1]) / res as f32;
        let step_z = (bounds[5] - bounds[2]) / res as f32;

        let mut next_logits = vec![FLASH_INVALID_SENTINEL; next_size * next_size * next_size];

        let mut curr_mask = extract_near_surface_mask(&grid_logits, grid_size, config.mc_level);
        for idx in 0..curr_mask.len() {
            if grid_logits[idx].abs() < 0.95 {
                curr_mask[idx] = 1;
            }
        }

        let expand_num = if level_idx == resolutions.len() - 1 {
            0
        } else {
            1
        };
        for _ in 0..expand_num {
            curr_mask = dilate_mask(&curr_mask, grid_size);
            curr_mask = dilate_mask(&curr_mask, grid_size);
        }

        let mut next_index = vec![0u8; next_logits.len()];
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let idx = (z * grid_size + y) * grid_size + x;
                    if curr_mask[idx] == 0 {
                        continue;
                    }
                    let nx = x * 2;
                    let ny = y * 2;
                    let nz = z * 2;
                    if nx < next_size && ny < next_size && nz < next_size {
                        let nidx = (nz * next_size + ny) * next_size + nx;
                        next_index[nidx] = 1;
                    }
                }
            }
        }

        for _ in 0..(2 - expand_num) {
            next_index = dilate_mask(&next_index, next_size);
        }

        let coords = collect_coords(&next_index, next_size, bounds, step_x, step_y, step_z);
        decode_flash_points(
            &latents,
            vae,
            &coords,
            next_size,
            num_chunks,
            &mut next_logits,
        )?;

        grid_logits = next_logits;
        grid_size = next_size;
    }

    let octree_resolution = 1usize << octree_depth;
    let mut sdf_values = Vec::with_capacity(grid_logits.len());
    for value in grid_logits {
        if value <= FLASH_INVALID_THRESHOLD {
            sdf_values.push(f32::NAN);
        } else {
            let sdf = -value / octree_resolution as f32;
            sdf_values.push(sdf);
        }
    }

    Ok(DenseGrid {
        values: sdf_values,
        size: [grid_size, grid_size, grid_size],
        bounds,
    })
}

fn build_flash_resolutions(
    octree_depth: usize,
    min_resolution: usize,
    mini_grid_num: usize,
) -> Vec<usize> {
    let mut resolutions = Vec::new();
    let mut octree_resolution = 1usize << octree_depth;
    if octree_resolution < min_resolution {
        resolutions.push(octree_resolution);
    }
    while octree_resolution >= min_resolution {
        resolutions.push(octree_resolution);
        octree_resolution /= 2;
    }
    resolutions.reverse();
    if let Some(first) = resolutions.first_mut() {
        let adjusted = (((*first as f32) / mini_grid_num as f32).round() as isize
            * mini_grid_num as isize
            - 1)
        .max(2) as usize;
        *first = adjusted;
    }
    for i in 1..resolutions.len() {
        resolutions[i] = resolutions[0] * (1usize << i);
    }
    resolutions
}

fn eval_flash_base_grid<B: Backend>(
    latents: Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    num_chunks: usize,
    mini_grid_num: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let grid_size = xs.len();
    let mini_grid_num = mini_grid_num.max(1);
    if !grid_size.is_multiple_of(mini_grid_num) {
        return Err(format!(
            "flash base grid size {grid_size} not divisible by mini_grid_num {mini_grid_num}"
        )
        .into());
    }
    let mini_size = (grid_size / mini_grid_num).max(1);
    let points_per_block = mini_size * mini_size * mini_size;
    let mut blocks_per_batch = (num_chunks / points_per_block).max(1);
    let max_points = flash_max_points::<B>();
    if std::env::var("TRIPOSG_FLASH_DEBUG").is_ok() {
        eprintln!(
            "flash_extract_geometry[base]: points_per_block={points_per_block} num_chunks={num_chunks} max_points={max_points}"
        );
    }
    let max_blocks = (max_points / points_per_block).max(1);
    blocks_per_batch = blocks_per_batch.min(max_blocks);

    let mut grid_values = vec![0.0f32; grid_size * grid_size * grid_size];
    let device = latents.device();

    let mut blocks = Vec::with_capacity(mini_grid_num * mini_grid_num * mini_grid_num);
    for bx in 0..mini_grid_num {
        for by in 0..mini_grid_num {
            for bz in 0..mini_grid_num {
                blocks.push([bx, by, bz]);
            }
        }
    }

    let mut coords = Vec::with_capacity(blocks_per_batch * points_per_block * 3);
    let mut indices = Vec::with_capacity(blocks_per_batch * points_per_block);
    let mut start = 0usize;
    while start < blocks.len() {
        let end = (start + blocks_per_batch).min(blocks.len());
        let batch_blocks = &blocks[start..end];
        let batch = batch_blocks.len();

        coords.clear();
        indices.clear();
        coords.reserve(batch * points_per_block * 3);
        indices.reserve(batch * points_per_block);
        for &[bx, by, bz] in batch_blocks {
            let base_x = bx * mini_size;
            let base_y = by * mini_size;
            let base_z = bz * mini_size;
            for ix in 0..mini_size {
                let gx = base_x + ix;
                for iy in 0..mini_size {
                    let gy = base_y + iy;
                    for iz in 0..mini_size {
                        let gz = base_z + iz;
                        let idx = (gz * grid_size + gy) * grid_size + gx;
                        coords.push(xs[gx]);
                        coords.push(ys[gy]);
                        coords.push(zs[gz]);
                        indices.push(idx);
                    }
                }
            }
        }

        let coords_tensor = Tensor::<B, 1>::from_floats(coords.as_slice(), &device).reshape([
            batch as i32,
            points_per_block as i32,
            3,
        ]);
        let latents_batch = latents.clone().repeat_dim(0, batch);
        let decoded = vae.decode(coords_tensor, latents_batch, None);
        let data = tensor_to_vec_f32(decoded)
            .map_err(|err| format!("failed to decode flash base grid: {err}"))?;
        for (i, &idx) in indices.iter().enumerate() {
            grid_values[idx] = data[i];
        }

        start = end;
    }

    Ok(grid_values)
}

fn extract_near_surface_mask(values: &[f32], size: usize, alpha: f32) -> Vec<u8> {
    let mut mask = vec![0u8; size * size * size];
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = (z * size + y) * size + x;
                let val = values[idx] + alpha;
                if val <= FLASH_INVALID_THRESHOLD {
                    continue;
                }
                let sign = val.signum();
                let mut same = true;
                for (dx, dy, dz) in [
                    (-1, 0, 0),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let nx = (x as isize + dx).clamp(0, size as isize - 1) as usize;
                    let ny = (y as isize + dy).clamp(0, size as isize - 1) as usize;
                    let nz = (z as isize + dz).clamp(0, size as isize - 1) as usize;
                    let nidx = (nz * size + ny) * size + nx;
                    let mut nval = values[nidx] + alpha;
                    if nval <= FLASH_INVALID_THRESHOLD {
                        nval = val;
                    }
                    if nval.signum() != sign {
                        same = false;
                        break;
                    }
                }
                if !same {
                    mask[idx] = 1;
                }
            }
        }
    }
    mask
}

fn dilate_mask(mask: &[u8], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; mask.len()];
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let idx = (z * size + y) * size + x;
                if mask[idx] == 0 {
                    continue;
                }
                for dz in -1isize..=1 {
                    let nz = z as isize + dz;
                    if nz < 0 || nz >= size as isize {
                        continue;
                    }
                    for dy in -1isize..=1 {
                        let ny = y as isize + dy;
                        if ny < 0 || ny >= size as isize {
                            continue;
                        }
                        for dx in -1isize..=1 {
                            let nx = x as isize + dx;
                            if nx < 0 || nx >= size as isize {
                                continue;
                            }
                            let nidx = (nz as usize * size + ny as usize) * size + nx as usize;
                            out[nidx] = 1;
                        }
                    }
                }
            }
        }
    }
    out
}

fn collect_coords(
    mask: &[u8],
    size: usize,
    bounds: [f32; 6],
    step_x: f32,
    step_y: f32,
    step_z: f32,
) -> Vec<([usize; 3], [f32; 3])> {
    let mut coords = Vec::new();
    for z in 0..size {
        let wz = bounds[2] + step_z * z as f32;
        for y in 0..size {
            let wy = bounds[1] + step_y * y as f32;
            for x in 0..size {
                let idx = (z * size + y) * size + x;
                if mask[idx] == 0 {
                    continue;
                }
                let wx = bounds[0] + step_x * x as f32;
                coords.push(([x, y, z], [wx, wy, wz]));
            }
        }
    }
    coords
}

fn decode_flash_points<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[([usize; 3], [f32; 3])],
    size: usize,
    num_chunks: usize,
    output: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if coords.is_empty() {
        return Ok(());
    }
    let device = latents.device();
    let mut coord_buf = Vec::with_capacity(num_chunks * 3);
    let mut indices = Vec::with_capacity(num_chunks);
    let mut start = 0usize;
    while start < coords.len() {
        let end = (start + num_chunks).min(coords.len());
        let slice = &coords[start..end];
        coord_buf.clear();
        indices.clear();
        coord_buf.reserve(slice.len().saturating_mul(3));
        indices.reserve(slice.len());
        for (grid_idx, world) in slice {
            coord_buf.extend_from_slice(world);
            indices.push((grid_idx[2] * size + grid_idx[1]) * size + grid_idx[0]);
        }
        let coords_tensor = Tensor::<B, 1>::from_floats(coord_buf.as_slice(), &device).reshape([
            1,
            slice.len() as i32,
            3,
        ]);
        let decoded = vae.decode(coords_tensor, latents.clone(), None);
        let data = tensor_to_vec_f32(decoded)
            .map_err(|err| format!("failed to decode flash grid values: {err}"))?;
        for (i, &idx) in indices.iter().enumerate() {
            output[idx] = data[i];
        }
        start = end;
    }
    Ok(())
}

fn eval_flash_base_grid_gpu<B: Backend>(
    vae: &TripoSGVae<B>,
    latent_proj: &Tensor<B, 3>,
    kv_cache: &Tensor<B, 3>,
    bounds: [f32; 6],
    base_res: usize,
    num_chunks: usize,
    mini_grid_num: usize,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let grid_size = base_res + 1;
    let mini_grid_num = mini_grid_num.max(1);
    if !grid_size.is_multiple_of(mini_grid_num) {
        return Err(format!(
            "flash base grid size {grid_size} not divisible by mini_grid_num {mini_grid_num}"
        )
        .into());
    }

    let mini_size = (grid_size / mini_grid_num).max(1);
    let points_per_block = mini_size * mini_size * mini_size;
    let blocks_per_batch = (num_chunks / points_per_block).max(1);

    let device = latent_proj.device();
    let total = grid_size * grid_size * grid_size;
    let mut grid_logits = Tensor::<B, 1>::full([total], FLASH_INVALID_SENTINEL, &device);

    let local_grid: Tensor<B, 4, Int> =
        Tensor::<B, 3, Int>::cartesian_grid([mini_size, mini_size, mini_size], &device);
    let local_grid = local_grid
        .reshape([points_per_block, 3])
        .unsqueeze_dim::<3>(0);

    let step = [
        (bounds[3] - bounds[0]) / base_res as f32,
        (bounds[4] - bounds[1]) / base_res as f32,
        (bounds[5] - bounds[2]) / base_res as f32,
    ];
    let mut blocks = Vec::with_capacity(mini_grid_num * mini_grid_num * mini_grid_num);
    for bx in 0..mini_grid_num {
        for by in 0..mini_grid_num {
            for bz in 0..mini_grid_num {
                blocks.push([bx as i32, by as i32, bz as i32]);
            }
        }
    }

    let mut shared_cache = Some(kv_cache.clone());
    let mut start = 0usize;
    while start < blocks.len() {
        let end = (start + blocks_per_batch).min(blocks.len());
        let batch_blocks = &blocks[start..end];
        let batch = batch_blocks.len();

        let mut offsets = Vec::with_capacity(batch * 3);
        for &[bx, by, bz] in batch_blocks {
            offsets.extend_from_slice(&[
                bx * mini_size as i32,
                by * mini_size as i32,
                bz * mini_size as i32,
            ]);
        }
        let offsets = TensorData::new(offsets, [batch, 3]);
        let offsets = Tensor::<B, 2, Int>::from_ints(offsets, &device).unsqueeze_dim::<3>(1);
        let coords_idx = offsets + local_grid.clone();
        let coords_idx = coords_idx.reshape([batch * points_per_block, 3]);
        let coords_world = coords_to_world_2(coords_idx.clone(), bounds, step);
        let indices = coords_to_linear_indices_2(coords_idx, grid_size);

        if std::env::var("TRIPOSG_FLASH_DEBUG").is_ok()
            && start == 0
            && let Ok(coord_data) = tensor_to_vec_f32(coords_world.clone())
        {
            let mut min = [f32::INFINITY; 3];
            let mut max = [f32::NEG_INFINITY; 3];
            for chunk in coord_data.chunks_exact(3) {
                for axis in 0..3 {
                    min[axis] = min[axis].min(chunk[axis]);
                    max[axis] = max[axis].max(chunk[axis]);
                }
            }
            eprintln!(
                "flash_extract_geometry[base]: coords world min={:?} max={:?} batch={batch}",
                min, max
            );
        }

        decode_flash_points_gpu(
            vae,
            latent_proj,
            &mut shared_cache,
            coords_world,
            indices,
            num_chunks,
            &mut grid_logits,
        )?;
        start = end;
    }

    Ok(grid_logits.reshape([grid_size, grid_size, grid_size]))
}

fn coords_to_linear_indices_2<B: Backend>(
    coords: Tensor<B, 2, Int>,
    size: usize,
) -> Tensor<B, 1, Int> {
    let device = coords.device();
    let idx0 = Tensor::<B, 1, Int>::from_ints([0], &device);
    let idx1 = Tensor::<B, 1, Int>::from_ints([1], &device);
    let idx2 = Tensor::<B, 1, Int>::from_ints([2], &device);

    let x = coords.clone().select(1, idx0).squeeze_dim(1);
    let y = coords.clone().select(1, idx1).squeeze_dim(1);
    let z = coords.select(1, idx2).squeeze_dim(1);

    let stride_x = (size * size) as i32;
    let stride_y = size as i32;
    x.mul_scalar(stride_x) + y.mul_scalar(stride_y) + z
}

fn coords_to_world_2<B: Backend>(
    coords: Tensor<B, 2, Int>,
    bounds: [f32; 6],
    step: [f32; 3],
) -> Tensor<B, 2> {
    let device = coords.device();
    let step_tensor = Tensor::<B, 1>::from_floats(step, &device).reshape([1, 3]);
    let min_tensor =
        Tensor::<B, 1>::from_floats([bounds[0], bounds[1], bounds[2]], &device).reshape([1, 3]);
    coords.float().mul(step_tensor).add(min_tensor)
}

fn decode_flash_points_gpu<B: Backend>(
    vae: &TripoSGVae<B>,
    latent_proj: &Tensor<B, 3>,
    kv_cache: &mut Option<Tensor<B, 3>>,
    coords: Tensor<B, 2>,
    indices: Tensor<B, 1, Int>,
    num_chunks: usize,
    output: &mut Tensor<B, 1>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (coords, indices) = maybe_group_flash_coords(coords, indices)?;
    let total = coords.shape().dims::<2>()[0];
    if total == 0 {
        return Ok(());
    }

    let mut out = output.clone();
    let max_points = flash_max_points::<B>();
    let chunk_points = num_chunks.min(max_points);
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk_points).min(total);
        let coords_chunk = coords.clone().slice([start..end, 0..3]).unsqueeze_dim(0);
        #[allow(clippy::single_range_in_vec_init)]
        let indices_chunk = indices.clone().slice([start..end]);

        let (decoded, cache) = vae.decode_with_latent_projection(
            coords_chunk,
            latent_proj.clone(),
            kv_cache.take(),
            None,
        );
        *kv_cache = Some(cache);

        let values = decoded.reshape([end - start]);
        // Burn scatter uses sum reduction. To get overwrite semantics, scatter deltas.
        let current = out.clone().gather(0, indices_chunk.clone());
        let delta = values - current;
        out = out.scatter(0, indices_chunk, delta);
        start = end;
    }

    *output = out;
    Ok(())
}

type FlashCoords<B> = (Tensor<B, 2>, Tensor<B, 1, Int>);

fn maybe_group_flash_coords<B: Backend>(
    coords: Tensor<B, 2>,
    indices: Tensor<B, 1, Int>,
) -> Result<FlashCoords<B>, Box<dyn std::error::Error>> {
    if std::env::var("TRIPOSG_FLASH_GROUP").is_err() {
        return Ok((coords, indices));
    }

    let grid = std::env::var("TRIPOSG_FLASH_GROUP_GRID")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(6)
        .max(1);
    let device = coords.device();

    let coords_data = tensor_to_vec_f32(coords.clone())
        .map_err(|err| format!("failed to read flash coords for grouping: {err}"))?;
    let indices_data = tensor_to_vec_i32(indices.clone())
        .map_err(|err| format!("failed to read flash indices for grouping: {err}"))?;

    let count = indices_data.len();
    if count == 0 {
        return Ok((coords, indices));
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for i in 0..count {
        let base = i * 3;
        let x = coords_data[base];
        let y = coords_data[base + 1];
        let z = coords_data[base + 2];
        min[0] = min[0].min(x);
        min[1] = min[1].min(y);
        min[2] = min[2].min(z);
        max[0] = max[0].max(x);
        max[1] = max[1].max(y);
        max[2] = max[2].max(z);
    }

    let mut order: Vec<(u32, usize)> = Vec::with_capacity(count);
    let grid_f = grid as f32;
    let eps = 1e-6f32;
    for i in 0..count {
        let base = i * 3;
        let x = coords_data[base];
        let y = coords_data[base + 1];
        let z = coords_data[base + 2];

        let nx = ((x - min[0]) / (max[0] - min[0] + eps) * (grid_f - 0.001))
            .floor()
            .clamp(0.0, grid_f - 1.0) as u32;
        let ny = ((y - min[1]) / (max[1] - min[1] + eps) * (grid_f - 0.001))
            .floor()
            .clamp(0.0, grid_f - 1.0) as u32;
        let nz = ((z - min[2]) / (max[2] - min[2] + eps) * (grid_f - 0.001))
            .floor()
            .clamp(0.0, grid_f - 1.0) as u32;
        let key = nx * (grid as u32 * grid as u32) + ny * grid as u32 + nz;
        order.push((key, i));
    }
    order.sort_by_key(|(key, _)| *key);

    let mut coords_sorted = vec![0.0f32; coords_data.len()];
    let mut indices_sorted = vec![0i32; indices_data.len()];
    for (dst, (_, src)) in order.iter().enumerate() {
        let src_base = src * 3;
        let dst_base = dst * 3;
        coords_sorted[dst_base] = coords_data[src_base];
        coords_sorted[dst_base + 1] = coords_data[src_base + 1];
        coords_sorted[dst_base + 2] = coords_data[src_base + 2];
        indices_sorted[dst] = indices_data[*src];
    }

    let coords = Tensor::<B, 2>::from_data(TensorData::new(coords_sorted, [count, 3]), &device);
    let idx_data = TensorData::new(indices_sorted, [count]);
    let indices = Tensor::<B, 1, Int>::from_data(idx_data.convert::<i32>(), &device);
    Ok((coords, indices))
}

fn extract_near_surface_mask_gpu<B: Backend>(
    values: &Tensor<B, 3>,
    alpha: f32,
) -> Tensor<B, 3, Bool> {
    let val = values.clone().add_scalar(alpha);
    let valid_mask = val.clone().greater_elem(FLASH_INVALID_THRESHOLD);

    let left = shift_with_replicate(&val, 0, 1);
    let right = shift_with_replicate(&val, 0, -1);
    let back = shift_with_replicate(&val, 1, 1);
    let front = shift_with_replicate(&val, 1, -1);
    let down = shift_with_replicate(&val, 2, 1);
    let up = shift_with_replicate(&val, 2, -1);

    let left_valid = left.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let right_valid = right.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let back_valid = back.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let front_valid = front.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let down_valid = down.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let up_valid = up.clone().greater_elem(FLASH_INVALID_THRESHOLD);

    let left = left.mask_where(left_valid.bool_not(), val.clone());
    let right = right.mask_where(right_valid.bool_not(), val.clone());
    let back = back.mask_where(back_valid.bool_not(), val.clone());
    let front = front.mask_where(front_valid.bool_not(), val.clone());
    let down = down.mask_where(down_valid.bool_not(), val.clone());
    let up = up.mask_where(up_valid.bool_not(), val.clone());

    let sign = val.clone().sign();
    let same_sign = left
        .sign()
        .equal(sign.clone())
        .bool_and(right.sign().equal(sign.clone()))
        .bool_and(back.sign().equal(sign.clone()))
        .bool_and(front.sign().equal(sign.clone()))
        .bool_and(down.sign().equal(sign.clone()))
        .bool_and(up.sign().equal(sign));

    same_sign.bool_not().bool_and(valid_mask)
}

fn dilate_mask_gpu<B: Backend>(mask: Tensor<B, 3, Bool>) -> Tensor<B, 3, Bool> {
    // Separable 3x3x3 dilation to avoid Conv3d allocations for large grids.
    let mask = dilate_axis_bool(mask, 0);
    let mask = dilate_axis_bool(mask, 1);
    dilate_axis_bool(mask, 2)
}

fn dilate_axis_bool<B: Backend>(mask: Tensor<B, 3, Bool>, axis: usize) -> Tensor<B, 3, Bool> {
    let neg = shift_with_replicate_bool(&mask, axis, -1);
    let pos = shift_with_replicate_bool(&mask, axis, 1);
    mask.bool_or(neg).bool_or(pos)
}

fn shift_with_replicate_bool<B: Backend>(
    tensor: &Tensor<B, 3, Bool>,
    axis: usize,
    shift: isize,
) -> Tensor<B, 3, Bool> {
    if shift == 0 {
        return tensor.clone();
    }
    let [sx, sy, sz] = tensor.shape().dims();
    let size = match axis {
        0 => sx,
        1 => sy,
        2 => sz,
        _ => unreachable!(),
    };
    if size <= 1 {
        return tensor.clone();
    }

    if shift > 0 {
        let main = slice_axis_bool(tensor, axis, 1, size);
        let tail = slice_axis_bool(tensor, axis, size - 1, size);
        Tensor::cat(vec![main, tail], axis)
    } else {
        let head = slice_axis_bool(tensor, axis, 0, 1);
        let main = slice_axis_bool(tensor, axis, 0, size - 1);
        Tensor::cat(vec![head, main], axis)
    }
}

fn slice_axis_bool<B: Backend>(
    tensor: &Tensor<B, 3, Bool>,
    axis: usize,
    start: usize,
    end: usize,
) -> Tensor<B, 3, Bool> {
    let [sx, sy, sz] = tensor.shape().dims();
    match axis {
        0 => tensor.clone().slice([start..end, 0..sy, 0..sz]),
        1 => tensor.clone().slice([0..sx, start..end, 0..sz]),
        2 => tensor.clone().slice([0..sx, 0..sy, start..end]),
        _ => unreachable!(),
    }
}

fn shift_with_replicate<B: Backend>(
    tensor: &Tensor<B, 3>,
    axis: usize,
    shift: isize,
) -> Tensor<B, 3> {
    if shift == 0 {
        return tensor.clone();
    }
    let [sx, sy, sz] = tensor.shape().dims();
    let size = match axis {
        0 => sx,
        1 => sy,
        2 => sz,
        _ => unreachable!(),
    };
    if size <= 1 {
        return tensor.clone();
    }

    if shift > 0 {
        let main = slice_axis(tensor, axis, 1, size);
        let tail = slice_axis(tensor, axis, size - 1, size);
        Tensor::cat(vec![main, tail], axis)
    } else {
        let head = slice_axis(tensor, axis, 0, 1);
        let main = slice_axis(tensor, axis, 0, size - 1);
        Tensor::cat(vec![head, main], axis)
    }
}

fn slice_axis<B: Backend>(
    tensor: &Tensor<B, 3>,
    axis: usize,
    start: usize,
    end: usize,
) -> Tensor<B, 3> {
    let [sx, sy, sz] = tensor.shape().dims();
    match axis {
        0 => tensor.clone().slice([start..end, 0..sy, 0..sz]),
        1 => tensor.clone().slice([0..sx, start..end, 0..sz]),
        2 => tensor.clone().slice([0..sx, 0..sy, start..end]),
        _ => unreachable!(),
    }
}

fn pow2(exp: usize) -> usize {
    1usize << exp
}

fn linspace(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start];
    }
    let step = (end - start) / (steps as f32 - 1.0);
    (0..steps).map(|i| start + step * i as f32).collect()
}

fn eval_grid<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    xs: &[f32],
    ys: &[f32],
    zs: &[f32],
    chunk_size: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let size = xs.len();
    let total = size * size * size;
    let mut values = vec![0.0f32; total];
    let device = latents.device();

    let mut coords = Vec::with_capacity(chunk_size * 3);
    let mut chunk_start = 0usize;

    for &zv in zs.iter() {
        for &yv in ys.iter() {
            for &xv in xs.iter() {
                coords.push(xv);
                coords.push(yv);
                coords.push(zv);
                let count = coords.len() / 3;
                if count >= chunk_size {
                    let end = chunk_start + count;
                    write_decoded_contiguous(
                        latents,
                        vae,
                        &coords,
                        &device,
                        &mut values[chunk_start..end],
                    )?;
                    coords.clear();
                    chunk_start = end;
                }
            }
        }
    }

    if !coords.is_empty() {
        let count = coords.len() / 3;
        let end = chunk_start + count;
        write_decoded_contiguous(
            latents,
            vae,
            &coords,
            &device,
            &mut values[chunk_start..end],
        )?;
    }

    Ok(values)
}

fn write_decoded_contiguous<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[f32],
    device: &B::Device,
    output_slice: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let count = coords.len() / 3;
    if count == 0 {
        return Ok(());
    }
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device).reshape([1, count as i32, 3]);
    let decoded = vae.decode(coords_tensor, latents.clone(), None);
    let data =
        tensor_to_vec_f32(decoded).map_err(|err| format!("failed to decode grid values: {err}"))?;
    output_slice.copy_from_slice(&data[..output_slice.len()]);
    Ok(())
}

fn write_decoded<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[f32],
    indices: &[usize],
    device: &B::Device,
    output: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let count = coords.len() / 3;
    if count == 0 {
        return Ok(());
    }
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device).reshape([1, count as i32, 3]);
    let decoded = vae.decode(coords_tensor, latents.clone(), None);
    let data =
        tensor_to_vec_f32(decoded).map_err(|err| format!("failed to decode grid values: {err}"))?;
    for (i, &dst) in indices.iter().enumerate() {
        output[dst] = data[i];
    }
    Ok(())
}

fn upsample_nearest(values: &[f32], size: usize) -> Vec<f32> {
    let next_size = size * 2;
    let mut out = vec![0.0f32; next_size * next_size * next_size];
    for z in 0..size {
        for y in 0..size {
            for x in 0..size {
                let val = values[(z * size + y) * size + x];
                let base_x = x * 2;
                let base_y = y * 2;
                let base_z = z * 2;
                for dz in 0..2 {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let nx = base_x + dx;
                            let ny = base_y + dy;
                            let nz = base_z + dz;
                            out[(nz * next_size + ny) * next_size + nx] = val;
                        }
                    }
                }
            }
        }
    }
    out
}

fn find_candidates_band(values: &[f32], size: usize, band_threshold: f32) -> Vec<[usize; 3]> {
    if size < 3 {
        return Vec::new();
    }
    if band_threshold >= 1.0 {
        let mut coords = Vec::with_capacity((size - 2) * (size - 2) * (size - 2));
        for z in 1..(size - 1) {
            for y in 1..(size - 1) {
                for x in 1..(size - 1) {
                    coords.push([x, y, z]);
                }
            }
        }
        return coords;
    }

    if band_threshold <= 0.0 {
        return Vec::new();
    }

    let (lower, upper) = band_threshold_bounds(band_threshold);
    let mut coords = Vec::new();
    for z in 1..(size - 1) {
        for y in 1..(size - 1) {
            for x in 1..(size - 1) {
                let idx = (z * size + y) * size + x;
                let logit = values[idx];
                if logit > lower && logit < upper {
                    coords.push([x, y, z]);
                }
            }
        }
    }
    coords
}

fn band_threshold_bounds(band_threshold: f32) -> (f32, f32) {
    let lower = (1.0 - band_threshold) * 0.5;
    let upper = (1.0 + band_threshold) * 0.5;
    let eps = 1e-6;
    let lower = lower.clamp(eps, 1.0 - eps);
    let upper = upper.clamp(eps, 1.0 - eps);
    let lower_logit = (lower / (1.0 - lower)).ln();
    let upper_logit = (upper / (1.0 - upper)).ln();
    (lower_logit, upper_logit)
}

fn expand_edge_region(coords: &[[usize; 3]], low_size: usize, high_size: usize) -> Vec<[usize; 3]> {
    if coords.is_empty() {
        return Vec::new();
    }
    let radius = if low_size < 512 { 2 } else { 1 };
    let dilated = dilate_coords(coords, low_size, radius);
    let mut out = Vec::new();
    let mut mask = vec![0u8; high_size * high_size * high_size];
    for coord in dilated {
        let base_x = coord[0] * 2;
        let base_y = coord[1] * 2;
        let base_z = coord[2] * 2;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let nx = base_x + dx;
                    let ny = base_y + dy;
                    let nz = base_z + dz;
                    if nx >= high_size || ny >= high_size || nz >= high_size {
                        continue;
                    }
                    let idx = (nz * high_size + ny) * high_size + nx;
                    if mask[idx] == 0 {
                        mask[idx] = 1;
                        out.push([nx, ny, nz]);
                    }
                }
            }
        }
    }
    out
}

fn dilate_coords(coords: &[[usize; 3]], size: usize, radius: usize) -> Vec<[usize; 3]> {
    let mut out = Vec::new();
    let mut mask = vec![0u8; size * size * size];
    let r = radius as isize;
    for coord in coords {
        let x = coord[0] as isize;
        let y = coord[1] as isize;
        let z = coord[2] as isize;
        for dz in -r..=r {
            let nz = z + dz;
            if nz < 0 || nz >= size as isize {
                continue;
            }
            for dy in -r..=r {
                let ny = y + dy;
                if ny < 0 || ny >= size as isize {
                    continue;
                }
                for dx in -r..=r {
                    let nx = x + dx;
                    if nx < 0 || nx >= size as isize {
                        continue;
                    }
                    let nxu = nx as usize;
                    let nyu = ny as usize;
                    let nzu = nz as usize;
                    let idx = (nzu * size + nyu) * size + nxu;
                    if mask[idx] == 0 {
                        mask[idx] = 1;
                        out.push([nxu, nyu, nzu]);
                    }
                }
            }
        }
    }
    out
}

fn update_grid_from_coords<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    coords: &[[usize; 3]],
    size: usize,
    bounds: [f32; 6],
    chunk_size: usize,
    grid: &mut [f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let device = latents.device();
    let mut buffer = Vec::with_capacity(chunk_size * 3);
    let mut indices = Vec::with_capacity(chunk_size);

    for coord in coords {
        let x = coord_to_world(coord[0], size, bounds[0], bounds[3]);
        let y = coord_to_world(coord[1], size, bounds[1], bounds[4]);
        let z = coord_to_world(coord[2], size, bounds[2], bounds[5]);
        buffer.push(x);
        buffer.push(y);
        buffer.push(z);
        indices.push((coord[2] * size + coord[1]) * size + coord[0]);

        if indices.len() >= chunk_size {
            write_decoded(latents, vae, &buffer, &indices, &device, grid)?;
            buffer.clear();
            indices.clear();
        }
    }

    if !indices.is_empty() {
        write_decoded(latents, vae, &buffer, &indices, &device, grid)?;
    }

    Ok(())
}

fn coord_to_world(coord: usize, size: usize, min: f32, max: f32) -> f32 {
    let center = (min + max) * 0.5;
    let half = (max - min) * 0.5;
    let offset = size as f32 / 2.0;
    center + (coord as f32 - offset) * (half / offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn in_band_sigmoid(logit: f32, band_threshold: f32) -> bool {
        let sdf = sigmoid(logit) * 2.0 - 1.0;
        sdf.abs() < band_threshold
    }

    #[test]
    fn band_threshold_bounds_match_sigmoid() {
        let thresholds = [0.1, 0.25, 0.5, 0.9];
        let logits = [-10.0, -2.0, -1.0, -0.1, 0.0, 0.2, 0.9, 2.0, 10.0];
        for &threshold in &thresholds {
            let (lower, upper) = band_threshold_bounds(threshold);
            for &logit in &logits {
                let via_bounds = logit > lower && logit < upper;
                let via_sigmoid = in_band_sigmoid(logit, threshold);
                assert_eq!(
                    via_bounds, via_sigmoid,
                    "mismatch for logit {logit} threshold {threshold}"
                );
            }
        }
    }
}

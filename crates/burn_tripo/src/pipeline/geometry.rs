use burn::prelude::*;
use burn::tensor::{IndexingUpdateOp, TensorData};

use crate::model::triposg::vae::TripoSGVae;
use crate::pipeline::mesh::DenseGrid;
use crate::readback::tensor_to_vec_f32;

const FLASH_INVALID_SENTINEL: f32 = -10000.0;
const FLASH_INVALID_THRESHOLD: f32 = -9000.0;
const FLASH_WGPU_MAX_POINTS: usize = 8192;
const FLASH_DEBUG: bool = false;

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
    latents: &Tensor<B, 3>,
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

    let mut grid_values = eval_grid(latents, vae, &xs, &ys, &zs, chunk_size)?;

    for depth in (dense_depth + 1)..=config.hierarchical_octree_depth {
        let next_size = pow2(depth);
        let mut high_values = upsample_nearest(&grid_values, size);

        let edge_coords = find_candidates_band(&grid_values, size, config.band_threshold);
        if !edge_coords.is_empty() {
            let expanded = expand_edge_region(&edge_coords, size, next_size);
            if !expanded.is_empty() {
                update_grid_from_coords(
                    latents,
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

#[cfg(not(target_arch = "wasm32"))]
pub fn flash_extract_geometry<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    if should_use_gpu_flash::<B>() {
        return flash_extract_geometry_gpu(latents, vae, config);
    }
    flash_extract_geometry_cpu(latents, vae, config)
}

#[cfg(target_arch = "wasm32")]
pub fn flash_extract_geometry<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let _ = (latents, vae, config);
    Err(
        "flash_extract_geometry sync path is unsupported on wasm; use async wasm flash extraction"
            .into(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn should_use_gpu_flash<B: Backend>() -> bool {
    let backend_name = std::any::type_name::<B>().to_ascii_lowercase();
    !backend_name.contains("ndarray")
}

#[cfg(not(target_arch = "wasm32"))]
type FlashCoord = ([usize; 3], [f32; 3]);

struct FlashRuntimePlan {
    bounds: [f32; 6],
    octree_depth: usize,
    num_chunks: usize,
    resolutions: Vec<usize>,
}

struct FlashGpuExtractState<B: Backend> {
    grid_logits: Tensor<B, 3>,
    grid_size: usize,
    bounds: [f32; 6],
    octree_depth: usize,
}

struct FlashRefinementLevelSetup<B: Backend> {
    next_size: usize,
    step: [f32; 3],
    next_logits: Tensor<B, 1>,
    curr_mask: Tensor<B, 3, Bool>,
    dilation_iters: usize,
}

struct FlashRefinementCoords<B: Backend> {
    curr_count: usize,
    next_coords: Tensor<B, 2, Int>,
    flat_indices: Tensor<B, 1, Int>,
}

fn build_flash_runtime_plan(config: &FlashExtractConfig) -> Result<FlashRuntimePlan, String> {
    let bounds = config.bounds;
    let octree_depth = config.octree_depth.max(1);
    let num_chunks = config.num_chunks.max(1);
    let min_resolution = config.min_resolution.max(2);
    let mini_grid_num = config.mini_grid_num.max(1);
    let resolutions = build_flash_resolutions(octree_depth, min_resolution, mini_grid_num);
    if resolutions.is_empty() {
        return Err("flash extractor produced empty resolution list".to_string());
    }
    Ok(FlashRuntimePlan {
        bounds,
        octree_depth,
        num_chunks,
        resolutions,
    })
}

fn flash_refinement_expand_num(level_idx: usize, levels_len: usize) -> usize {
    if level_idx + 1 == levels_len { 0 } else { 1 }
}

fn build_flash_refinement_curr_mask_gpu<B: Backend>(
    grid_logits: &Tensor<B, 3>,
    mc_level: f32,
    expand_num: usize,
) -> Tensor<B, 3, Bool> {
    let mut curr_mask = extract_near_surface_mask_gpu(grid_logits, mc_level);
    let near_mask = grid_logits.clone().abs().lower_elem(0.95);
    curr_mask = curr_mask.bool_or(near_mask);
    for _ in 0..expand_num {
        curr_mask = dilate_mask_gpu(curr_mask);
        curr_mask = dilate_mask_gpu(curr_mask);
    }
    curr_mask
}

fn prepare_flash_refinement_level_gpu<B: Backend>(
    grid_logits: &Tensor<B, 3>,
    plan: &FlashRuntimePlan,
    config: &FlashExtractConfig,
    level_idx: usize,
    res: usize,
) -> FlashRefinementLevelSetup<B> {
    let next_size = res + 1;
    let step = [
        (plan.bounds[3] - plan.bounds[0]) / res as f32,
        (plan.bounds[4] - plan.bounds[1]) / res as f32,
        (plan.bounds[5] - plan.bounds[2]) / res as f32,
    ];
    let device = grid_logits.device();
    let next_logits = Tensor::<B, 1>::full(
        [next_size * next_size * next_size],
        FLASH_INVALID_SENTINEL,
        &device,
    );
    let expand_num = flash_refinement_expand_num(level_idx, plan.resolutions.len());
    let curr_mask = build_flash_refinement_curr_mask_gpu(grid_logits, config.mc_level, expand_num);

    FlashRefinementLevelSetup {
        next_size,
        step,
        next_logits,
        curr_mask,
        dilation_iters: 2 - expand_num,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn build_flash_refinement_step(
    grid_logits: &[f32],
    grid_size: usize,
    bounds: [f32; 6],
    res: usize,
    is_last_level: bool,
    mc_level: f32,
) -> (usize, Vec<f32>, Vec<FlashCoord>) {
    let next_size = res + 1;
    let step_x = (bounds[3] - bounds[0]) / res as f32;
    let step_y = (bounds[4] - bounds[1]) / res as f32;
    let step_z = (bounds[5] - bounds[2]) / res as f32;

    let next_logits = vec![FLASH_INVALID_SENTINEL; next_size * next_size * next_size];
    let mut curr_mask = extract_near_surface_mask(grid_logits, grid_size, mc_level);
    for idx in 0..curr_mask.len() {
        if grid_logits[idx].abs() < 0.95 {
            curr_mask[idx] = 1;
        }
    }

    let expand_num = if is_last_level { 0 } else { 1 };
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
    (next_size, next_logits, coords)
}

#[cfg(not(target_arch = "wasm32"))]
fn finalize_flash_logits_to_sdf(mut grid_logits: Vec<f32>, octree_depth: usize) -> Vec<f32> {
    let octree_resolution = 1usize << octree_depth;
    for value in &mut grid_logits {
        if *value <= FLASH_INVALID_THRESHOLD {
            *value = f32::NAN;
        } else {
            *value = -*value / octree_resolution as f32;
        }
    }
    grid_logits
}

fn finalize_flash_logits_tensor<B: Backend>(
    grid_logits: Tensor<B, 3>,
    octree_depth: usize,
) -> Tensor<B, 3> {
    let invalid = grid_logits
        .clone()
        .lower_equal_elem(FLASH_INVALID_THRESHOLD);
    let nan = Tensor::<B, 1>::from_floats([f32::NAN], &grid_logits.device()).reshape([1, 1, 1]);
    let logits = grid_logits.mask_where(invalid, nan);
    let octree_resolution = 1usize << octree_depth;
    logits.mul_scalar(-1.0 / octree_resolution as f32)
}

#[cfg(target_arch = "wasm32")]
pub async fn flash_extract_geometry_async_wasm<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, String> {
    let state = flash_extract_geometry_gpu_shared_async_wasm(latents, vae, config)
        .await
        .map_err(|err| format!("flash extraction failed: {err}"))?;
    let sdf = finalize_flash_logits_tensor(state.grid_logits, state.octree_depth);
    let sdf_values = tensor_to_vec_f32_async_wasm(sdf)
        .await
        .map_err(|err| format!("failed to read flash grid logits: {err}"))?;

    Ok(DenseGrid {
        values: sdf_values,
        size: [state.grid_size, state.grid_size, state.grid_size],
        bounds: state.bounds,
    })
}

#[cfg(target_arch = "wasm32")]
async fn flash_extract_geometry_gpu_shared_async_wasm<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<FlashGpuExtractState<B>, Box<dyn std::error::Error>> {
    let plan = build_flash_runtime_plan(config).map_err(std::io::Error::other)?;
    let base_res = plan.resolutions[0];
    let base_grid = base_res + 1;

    let latent_proj = vae.prepare_latent_projection(latents, None);
    let initial_kv_cache = Some(vae.build_kv_cache(latent_proj.clone(), None));
    let mut grid_logits = eval_flash_base_grid_gpu(
        vae,
        &latent_proj,
        initial_kv_cache.clone(),
        plan.bounds,
        base_res,
        plan.num_chunks,
        config.mini_grid_num.max(1),
    )?;
    web_sys::console::log_1(
        &format!(
            "burn_synth wasm flash: base grid ready (base_res={} grid={})",
            base_res, base_grid
        )
        .into(),
    );
    let mut grid_size = base_grid;
    let mut shared_kv_cache = initial_kv_cache;

    for (level_idx, &res) in plan.resolutions.iter().enumerate().skip(1) {
        let FlashRefinementLevelSetup {
            next_size,
            step,
            mut next_logits,
            curr_mask,
            dilation_iters,
        } = prepare_flash_refinement_level_gpu(&grid_logits, &plan, config, level_idx, res);
        let device = grid_logits.device();

        let FlashRefinementCoords {
            curr_count,
            next_coords,
            flat_indices,
        } = build_flash_refinement_coords_from_mask_async_wasm(
            curr_mask,
            grid_size,
            next_size,
            dilation_iters,
            &device,
        )
        .await
        .map_err(std::io::Error::other)?;
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm flash: level={} curr_count={} grid_size={} next_res={}",
                level_idx, curr_count, grid_size, next_size
            )
            .into(),
        );
        if curr_count == 0 {
            break;
        }
        let next_count = next_coords.shape().dims::<2>()[0];
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm flash: level={} next_count={} next_size={}",
                level_idx, next_count, next_size
            )
            .into(),
        );
        if next_count == 0 {
            break;
        }

        let world_coords = coords_to_world_2(&next_coords, plan.bounds, step);
        drop(next_coords);

        decode_flash_points_gpu(
            vae,
            &latent_proj,
            &mut shared_kv_cache,
            world_coords,
            flat_indices,
            plan.num_chunks,
            &mut next_logits,
        )?;
        web_sys::console::log_1(
            &format!(
                "burn_synth wasm flash: level={} decode_done next_size={}",
                level_idx, next_size
            )
            .into(),
        );

        grid_logits = next_logits.reshape([next_size as i32, next_size as i32, next_size as i32]);
        grid_size = next_size;
    }

    Ok(FlashGpuExtractState {
        grid_logits,
        grid_size,
        bounds: plan.bounds,
        octree_depth: plan.octree_depth,
    })
}

#[cfg(target_arch = "wasm32")]
async fn tensor_to_vec_f32_async_wasm<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Result<Vec<f32>, String> {
    tensor
        .into_data_async()
        .await
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("failed to read tensor data: {err:?}"))
}

#[cfg(target_arch = "wasm32")]
async fn build_flash_refinement_coords_from_mask_async_wasm<B: Backend>(
    curr_mask: Tensor<B, 3, Bool>,
    grid_size: usize,
    next_size: usize,
    dilation_iters: usize,
    device: &B::Device,
) -> Result<FlashRefinementCoords<B>, String> {
    let curr_mask_values = curr_mask
        .into_data_async()
        .await
        .convert::<bool>()
        .to_vec::<bool>()
        .map_err(|err| format!("failed to read refinement mask: {err:?}"))?;
    let (curr_count, flat_indices_host, next_coords_host) = build_flash_refinement_compact_host(
        curr_mask_values.as_slice(),
        grid_size,
        next_size,
        dilation_iters,
    )?;

    let next_count = flat_indices_host.len();
    let flat_indices =
        Tensor::<B, 1, Int>::from_data(TensorData::new(flat_indices_host, [next_count]), device);
    let next_coords =
        Tensor::<B, 1, Int>::from_data(TensorData::new(next_coords_host, [next_count * 3]), device)
            .reshape([next_count, 3]);

    Ok(FlashRefinementCoords {
        curr_count,
        next_coords,
        flat_indices,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn build_flash_refinement_coords_from_mask_gpu<B: Backend>(
    curr_mask: Tensor<B, 3, Bool>,
    next_total: usize,
    next_size: usize,
    dilation_iters: usize,
    device: &B::Device,
) -> FlashRefinementCoords<B> {
    let curr_coords = curr_mask.argwhere();
    let curr_count = curr_coords.shape().dims::<2>()[0];
    let doubled = curr_coords.mul_scalar(2);
    let doubled_indices = coords_to_linear_indices_2(&doubled, next_size);
    drop(doubled);
    let mut next_index = flash_refinement_next_index_mask_from_doubled_indices(
        doubled_indices,
        next_total,
        next_size,
        device,
    );
    for _ in 0..dilation_iters {
        next_index = dilate_mask_gpu(next_index);
    }
    let next_coords = next_index.argwhere();
    let flat_indices = coords_to_linear_indices_2(&next_coords, next_size);
    FlashRefinementCoords {
        curr_count,
        next_coords,
        flat_indices,
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn build_flash_refinement_compact_host(
    curr_mask: &[bool],
    grid_size: usize,
    next_size: usize,
    dilation_iters: usize,
) -> Result<(usize, Vec<i32>, Vec<i32>), String> {
    let grid_plane = grid_size
        .checked_mul(grid_size)
        .ok_or_else(|| "refinement grid plane size overflow".to_string())?;
    let grid_total = grid_plane
        .checked_mul(grid_size)
        .ok_or_else(|| "refinement grid size overflow".to_string())?;
    if curr_mask.len() != grid_total {
        return Err(format!(
            "refinement mask size mismatch: got {}, expected {} (grid_size={grid_size})",
            curr_mask.len(),
            grid_total
        ));
    }
    let next_plane = next_size
        .checked_mul(next_size)
        .ok_or_else(|| "refinement next-grid plane size overflow".to_string())?;
    let next_total = next_plane
        .checked_mul(next_size)
        .ok_or_else(|| "refinement next-grid size overflow".to_string())?;
    let next_max = next_size as isize - 1;
    let radius = dilation_iters as isize;

    let mut curr_count = 0usize;
    let mut next_mask = vec![0u8; next_total];

    for (idx, &active) in curr_mask.iter().enumerate() {
        if !active {
            continue;
        }
        curr_count += 1;
        let z = idx / grid_plane;
        let rem = idx - z * grid_plane;
        let y = rem / grid_size;
        let x = rem - y * grid_size;

        let base_x = (x * 2) as isize;
        let base_y = (y * 2) as isize;
        let base_z = (z * 2) as isize;

        for dz in -radius..=radius {
            let nz = (base_z + dz).clamp(0, next_max) as usize;
            let z_off = nz * next_plane;
            for dy in -radius..=radius {
                let ny = (base_y + dy).clamp(0, next_max) as usize;
                let yz_off = z_off + ny * next_size;
                for dx in -radius..=radius {
                    let nx = (base_x + dx).clamp(0, next_max) as usize;
                    next_mask[yz_off + nx] = 1;
                }
            }
        }
    }

    let mut flat_indices = Vec::new();
    let mut coords = Vec::new();
    for (linear, &active) in next_mask.iter().enumerate() {
        if active == 0 {
            continue;
        }
        let linear_i32 =
            i32::try_from(linear).map_err(|_| "refinement index exceeds i32::MAX".to_string())?;
        let z = linear / next_plane;
        let rem = linear - z * next_plane;
        let y = rem / next_size;
        let x = rem - y * next_size;
        let x_i32 =
            i32::try_from(x).map_err(|_| "refinement x coordinate exceeds i32::MAX".to_string())?;
        let y_i32 =
            i32::try_from(y).map_err(|_| "refinement y coordinate exceeds i32::MAX".to_string())?;
        let z_i32 =
            i32::try_from(z).map_err(|_| "refinement z coordinate exceeds i32::MAX".to_string())?;
        flat_indices.push(linear_i32);
        coords.extend_from_slice(&[x_i32, y_i32, z_i32]);
    }

    Ok((curr_count, flat_indices, coords))
}

fn flash_max_points<B: Backend>() -> usize {
    if is_wgpu_backend::<B>() {
        return FLASH_WGPU_MAX_POINTS;
    }
    usize::MAX
}

fn is_wgpu_backend<B: Backend>() -> bool {
    std::any::type_name::<B>()
        .to_ascii_lowercase()
        .contains("wgpu")
}

fn needs_wasm_f16_alignment_workaround<B: Backend>() -> bool {
    cfg!(target_arch = "wasm32") && std::mem::size_of::<B::FloatElem>() == 2
}

fn flash_blocks_per_batch<B: Backend>(num_chunks: usize, points_per_block: usize) -> usize {
    let requested = num_chunks.div_ceil(points_per_block).max(1);
    let max_points = flash_max_points::<B>();
    if FLASH_DEBUG {
        eprintln!(
            "flash_extract_geometry[base]: points_per_block={points_per_block} num_chunks={num_chunks} max_points={max_points}"
        );
    }
    let max_blocks = (max_points / points_per_block).max(1);
    requested.min(max_blocks)
}

#[cfg(not(target_arch = "wasm32"))]
fn flash_refinement_next_index_mask_from_doubled_indices<B: Backend>(
    doubled_indices: Tensor<B, 1, Int>,
    next_total: usize,
    next_size: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let ones = Tensor::<B, 1>::ones([doubled_indices.shape().dims::<1>()[0]], device);
    let mut next_index = Tensor::<B, 1>::zeros([next_total], device);
    next_index = next_index.scatter(0, doubled_indices, ones, IndexingUpdateOp::Add);
    next_index
        .reshape([next_size as i32, next_size as i32, next_size as i32])
        .greater_elem(0.0)
}

fn flash_base_blocks(mini_grid_num: usize) -> Vec<[usize; 3]> {
    let mut blocks = Vec::with_capacity(mini_grid_num * mini_grid_num * mini_grid_num);
    for bx in 0..mini_grid_num {
        for by in 0..mini_grid_num {
            for bz in 0..mini_grid_num {
                blocks.push([bx, by, bz]);
            }
        }
    }
    blocks
}

#[cfg(not(target_arch = "wasm32"))]
fn fill_flash_base_batch_buffers(
    batch_blocks: &[[usize; 3]],
    mini_size: usize,
    grid_size: usize,
    axes: [&[f32]; 3],
    coords: &mut Vec<f32>,
    indices: &mut Vec<usize>,
) {
    coords.clear();
    indices.clear();
    let batch_len = batch_blocks.len();
    let points_per_block = mini_size * mini_size * mini_size;
    coords.reserve(batch_len * points_per_block * 3);
    indices.reserve(batch_len * points_per_block);
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
                    coords.push(axes[0][gx]);
                    coords.push(axes[1][gy]);
                    coords.push(axes[2][gz]);
                    indices.push(idx);
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn fill_flash_refinement_batch_buffers(
    coords_src: &[FlashCoord],
    size: usize,
    coords: &mut Vec<f32>,
    indices: &mut Vec<usize>,
) {
    coords.clear();
    indices.clear();
    coords.reserve(coords_src.len().saturating_mul(3));
    indices.reserve(coords_src.len());
    for (grid_idx, world) in coords_src {
        coords.extend_from_slice(world);
        indices.push((grid_idx[2] * size + grid_idx[1]) * size + grid_idx[0]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn write_flash_batch_values(
    output: &mut [f32],
    indices: &[usize],
    decoded: &[f32],
    stage: &str,
) -> Result<(), String> {
    if decoded.len() < indices.len() {
        return Err(format!(
            "{stage} decoded {} values for {} points",
            decoded.len(),
            indices.len()
        ));
    }
    for (i, &idx) in indices.iter().enumerate() {
        output[idx] = decoded[i];
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_flash_base_batch<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    device: &B::Device,
    coords: &[f32],
    blocks_in_batch: usize,
    points_per_block: usize,
) -> Tensor<B, 3> {
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device).reshape([
        blocks_in_batch as i32,
        points_per_block as i32,
        3,
    ]);
    let latents_batch = latents.clone().repeat_dim(0, blocks_in_batch);
    vae.decode(coords_tensor, &latents_batch, None)
}

#[cfg(not(target_arch = "wasm32"))]
fn decode_flash_refinement_batch<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    device: &B::Device,
    coords: &[f32],
    points: usize,
) -> Tensor<B, 3> {
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device).reshape([1, points as i32, 3]);
    vae.decode(coords_tensor, latents, None)
}

#[cfg(not(target_arch = "wasm32"))]
fn flash_extract_geometry_gpu<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let state = flash_extract_geometry_gpu_shared(latents, vae, config)?;
    let sdf = finalize_flash_logits_tensor(state.grid_logits, state.octree_depth);
    let sdf_values =
        tensor_to_vec_f32(sdf).map_err(|err| format!("failed to read flash grid logits: {err}"))?;

    Ok(DenseGrid {
        values: sdf_values,
        size: [state.grid_size, state.grid_size, state.grid_size],
        bounds: state.bounds,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn flash_extract_geometry_gpu_shared<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<FlashGpuExtractState<B>, Box<dyn std::error::Error>> {
    let plan = build_flash_runtime_plan(config).map_err(std::io::Error::other)?;
    let base_res = plan.resolutions[0];
    let base_grid = base_res + 1;

    let latent_proj = vae.prepare_latent_projection(latents, None);
    let initial_kv_cache = Some(vae.build_kv_cache(latent_proj.clone(), None));

    let mut grid_logits = eval_flash_base_grid_gpu(
        vae,
        &latent_proj,
        initial_kv_cache.clone(),
        plan.bounds,
        base_res,
        plan.num_chunks,
        config.mini_grid_num.max(1),
    )?;
    let mut grid_size = base_grid;
    #[cfg(not(target_arch = "wasm32"))]
    log_flash_stats("base", &grid_logits, grid_size);
    let mut shared_kv_cache = initial_kv_cache;
    for (level_idx, &res) in plan.resolutions.iter().enumerate().skip(1) {
        let FlashRefinementLevelSetup {
            next_size,
            step,
            mut next_logits,
            curr_mask,
            dilation_iters,
        } = prepare_flash_refinement_level_gpu(&grid_logits, &plan, config, level_idx, res);
        let next_total = next_size * next_size * next_size;
        let device = grid_logits.device();
        let FlashRefinementCoords {
            curr_count,
            next_coords,
            flat_indices,
        } = build_flash_refinement_coords_from_mask_gpu(
            curr_mask,
            next_total,
            next_size,
            dilation_iters,
            &device,
        );
        if FLASH_DEBUG {
            eprintln!(
                "flash_extract_geometry[level={level_idx}] curr_count={curr_count} grid_size={grid_size} next_size={next_size}"
            );
        }
        if curr_count == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            log_flash_level_empty("curr_mask", level_idx, grid_size, next_size);
            break;
        }

        let next_count = next_coords.shape().dims::<2>()[0];
        if FLASH_DEBUG {
            eprintln!(
                "flash_extract_geometry[level={level_idx}] next_count={next_count} next_size={next_size}"
            );
        }
        if next_count == 0 {
            #[cfg(not(target_arch = "wasm32"))]
            log_flash_level_empty("next_mask", level_idx, grid_size, next_size);
            break;
        }

        let world_coords = coords_to_world_2(&next_coords, plan.bounds, step);
        drop(next_coords);

        decode_flash_points_gpu(
            vae,
            &latent_proj,
            &mut shared_kv_cache,
            world_coords,
            flat_indices,
            plan.num_chunks,
            &mut next_logits,
        )?;

        grid_logits = next_logits.reshape([next_size as i32, next_size as i32, next_size as i32]);
        grid_size = next_size;
        #[cfg(not(target_arch = "wasm32"))]
        log_flash_stats(&format!("level-{level_idx}"), &grid_logits, grid_size);
    }

    Ok(FlashGpuExtractState {
        grid_logits,
        grid_size,
        bounds: plan.bounds,
        octree_depth: plan.octree_depth,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn log_flash_stats<B: Backend>(label: &str, grid: &Tensor<B, 3>, size: usize) {
    if !FLASH_DEBUG {
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

#[cfg(not(target_arch = "wasm32"))]
fn log_flash_level_empty(reason: &str, level_idx: usize, curr: usize, next: usize) {
    if !FLASH_DEBUG {
        return;
    }
    eprintln!(
        "flash_extract_geometry: empty {reason} at level {level_idx} (curr={curr}, next={next}), stopping refinement"
    );
}

fn eval_flash_base_grid_gpu<B: Backend>(
    vae: &TripoSGVae<B>,
    latent_proj: &Tensor<B, 3>,
    kv_cache: Option<Tensor<B, 3>>,
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
    let blocks_per_batch = flash_blocks_per_batch::<B>(num_chunks, points_per_block);
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
    let blocks = flash_base_blocks(mini_grid_num);

    let mut shared_cache = kv_cache;
    for batch_blocks in blocks.chunks(blocks_per_batch) {
        let mut offsets = Vec::with_capacity(batch_blocks.len() * 3);
        for &[bx, by, bz] in batch_blocks {
            offsets.extend_from_slice(&[
                (bx * mini_size) as i32,
                (by * mini_size) as i32,
                (bz * mini_size) as i32,
            ]);
        }
        let offsets = TensorData::new(offsets, [batch_blocks.len(), 3]);
        let offsets = Tensor::<B, 2, Int>::from_ints(offsets, &device).unsqueeze_dim::<3>(1);
        let coords_idx = offsets + local_grid.clone();
        let coords_idx = coords_idx.reshape([batch_blocks.len() * points_per_block, 3]);
        let coords_world = coords_to_world_2(&coords_idx, bounds, step);
        let indices = coords_to_linear_indices_2(&coords_idx, grid_size);

        decode_flash_points_gpu(
            vae,
            latent_proj,
            &mut shared_cache,
            coords_world,
            indices,
            num_chunks,
            &mut grid_logits,
        )?;
    }

    Ok(grid_logits.reshape([grid_size, grid_size, grid_size]))
}

fn coords_to_linear_indices_2<B: Backend>(
    coords: &Tensor<B, 2, Int>,
    size: usize,
) -> Tensor<B, 1, Int> {
    let device = coords.device();
    let idx0 = Tensor::<B, 1, Int>::from_ints([0], &device);
    let idx1 = Tensor::<B, 1, Int>::from_ints([1], &device);
    let idx2 = Tensor::<B, 1, Int>::from_ints([2], &device);

    let x = coords.clone().select(1, idx0).squeeze_dim(1);
    let y = coords.clone().select(1, idx1).squeeze_dim(1);
    let z = coords.clone().select(1, idx2).squeeze_dim(1);

    let stride_z = (size * size) as i32;
    let stride_y = size as i32;
    z.mul_scalar(stride_z) + y.mul_scalar(stride_y) + x
}

fn coords_to_world_2<B: Backend>(
    coords: &Tensor<B, 2, Int>,
    bounds: [f32; 6],
    step: [f32; 3],
) -> Tensor<B, 2> {
    let device = coords.device();
    let step_tensor = Tensor::<B, 1>::from_floats(step, &device).reshape([1, 3]);
    let min_tensor =
        Tensor::<B, 1>::from_floats([bounds[0], bounds[1], bounds[2]], &device).reshape([1, 3]);
    coords.clone().float().mul(step_tensor).add(min_tensor)
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
    let max_points = flash_max_points::<B>().max(1);
    let chunk_points = num_chunks.max(1).min(max_points);
    let align_f16_chunks = needs_wasm_f16_alignment_workaround::<B>();
    let mut start = 0usize;
    while start < total {
        let end = (start + chunk_points).min(total);
        let valid_points = end - start;
        let mut coords_chunk = coords.clone().slice([start..end, 0..3]).unsqueeze_dim(0);
        #[allow(clippy::single_range_in_vec_init)]
        let mut indices_chunk = indices.clone().slice([start..end]);
        let needs_padding = align_f16_chunks && !valid_points.is_multiple_of(2);
        let decode_points = valid_points + usize::from(needs_padding);
        if needs_padding {
            let pad = coords_chunk
                .clone()
                .slice([0..1, (valid_points - 1)..valid_points, 0..3]);
            coords_chunk = Tensor::cat(vec![coords_chunk, pad], 1);
            #[allow(clippy::single_range_in_vec_init)]
            let pad_index = indices_chunk
                .clone()
                .slice([(valid_points - 1)..valid_points]);
            indices_chunk = Tensor::cat(vec![indices_chunk, pad_index], 0);
        }

        let (decoded, cache) = vae.decode_with_latent_projection(
            coords_chunk,
            latent_proj.clone(),
            kv_cache.take(),
            None,
        );
        *kv_cache = Some(cache);
        let values = decoded.reshape([decode_points]);
        let mut delta = values.add_scalar(-FLASH_INVALID_SENTINEL);
        if needs_padding {
            delta = zero_padded_flash_delta_tail(delta, valid_points);
        }
        // Burn scatter uses sum reduction. Flash indices are unique per decode pass.
        // For padded fp16 wasm chunks we append one extra index and zero out its delta.
        out = out.scatter(0, indices_chunk, delta, IndexingUpdateOp::Add);
        start = end;
    }

    *output = out;
    Ok(())
}

fn zero_padded_flash_delta_tail<B: Backend>(
    delta: Tensor<B, 1>,
    valid_points: usize,
) -> Tensor<B, 1> {
    let len = delta.shape().dims::<1>()[0];
    if valid_points >= len {
        return delta;
    }
    let keep =
        Tensor::<B, 1, Int>::arange(0..len as i64, &delta.device()).lower_elem(valid_points as i64);
    let zeros = Tensor::<B, 1>::zeros([1], &delta.device());
    delta.mask_where(keep.bool_not(), zeros)
}

type FlashCoords<B> = (Tensor<B, 2>, Tensor<B, 1, Int>);

fn maybe_group_flash_coords<B: Backend>(
    coords: Tensor<B, 2>,
    indices: Tensor<B, 1, Int>,
) -> Result<FlashCoords<B>, Box<dyn std::error::Error>> {
    Ok((coords, indices))
}

fn extract_near_surface_mask_gpu<B: Backend>(
    values: &Tensor<B, 3>,
    alpha: f32,
) -> Tensor<B, 3, Bool> {
    let val = values.clone().add_scalar(alpha);
    let valid_mask = val.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let sign = val.clone().sign();

    let mut same_sign = neighbor_sign_matches(&val, &sign, 0, 1);
    same_sign = same_sign.bool_and(neighbor_sign_matches(&val, &sign, 0, -1));
    same_sign = same_sign.bool_and(neighbor_sign_matches(&val, &sign, 1, 1));
    same_sign = same_sign.bool_and(neighbor_sign_matches(&val, &sign, 1, -1));
    same_sign = same_sign.bool_and(neighbor_sign_matches(&val, &sign, 2, 1));
    same_sign = same_sign.bool_and(neighbor_sign_matches(&val, &sign, 2, -1));

    same_sign.bool_not().bool_and(valid_mask)
}

fn neighbor_sign_matches<B: Backend>(
    values: &Tensor<B, 3>,
    center_sign: &Tensor<B, 3>,
    axis: usize,
    shift: isize,
) -> Tensor<B, 3, Bool> {
    let shifted = shift_with_replicate(values, axis, shift);
    let shifted_valid = shifted.clone().greater_elem(FLASH_INVALID_THRESHOLD);
    let shifted = shifted.mask_where(shifted_valid.bool_not(), values.clone());
    shifted.sign().equal(center_sign.clone())
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

#[cfg(not(target_arch = "wasm32"))]
fn flash_extract_geometry_cpu<B: Backend>(
    latents: &Tensor<B, 3>,
    vae: &TripoSGVae<B>,
    config: &FlashExtractConfig,
) -> Result<DenseGrid, Box<dyn std::error::Error>> {
    let plan = build_flash_runtime_plan(config).map_err(std::io::Error::other)?;
    let base_res = plan.resolutions[0];
    let base_grid = base_res + 1;
    let (xs, ys, zs) = (
        linspace(plan.bounds[0], plan.bounds[3], base_grid),
        linspace(plan.bounds[1], plan.bounds[4], base_grid),
        linspace(plan.bounds[2], plan.bounds[5], base_grid),
    );

    let mut grid_logits = eval_flash_base_grid(
        latents,
        vae,
        &xs,
        &ys,
        &zs,
        plan.num_chunks,
        config.mini_grid_num.max(1),
    )?;
    let mut grid_size = base_grid;

    for (level_idx, &res) in plan.resolutions.iter().enumerate().skip(1) {
        let is_last_level = level_idx == plan.resolutions.len() - 1;
        let (next_size, mut next_logits, coords) = build_flash_refinement_step(
            &grid_logits,
            grid_size,
            plan.bounds,
            res,
            is_last_level,
            config.mc_level,
        );
        decode_flash_points(
            latents,
            vae,
            &coords,
            next_size,
            plan.num_chunks,
            &mut next_logits,
        )?;

        grid_logits = next_logits;
        grid_size = next_size;
    }

    let sdf_values = finalize_flash_logits_to_sdf(grid_logits, plan.octree_depth);

    Ok(DenseGrid {
        values: sdf_values,
        size: [grid_size, grid_size, grid_size],
        bounds: plan.bounds,
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

#[cfg(not(target_arch = "wasm32"))]
fn eval_flash_base_grid<B: Backend>(
    latents: &Tensor<B, 3>,
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
    let blocks_per_batch = flash_blocks_per_batch::<B>(num_chunks, points_per_block);

    let mut grid_values = vec![0.0f32; grid_size * grid_size * grid_size];
    let device = latents.device();

    let blocks = flash_base_blocks(mini_grid_num);

    let mut coords = Vec::with_capacity(blocks_per_batch * points_per_block * 3);
    let mut indices = Vec::with_capacity(blocks_per_batch * points_per_block);
    for batch_blocks in blocks.chunks(blocks_per_batch) {
        fill_flash_base_batch_buffers(
            batch_blocks,
            mini_size,
            grid_size,
            [xs, ys, zs],
            &mut coords,
            &mut indices,
        );
        let decoded = decode_flash_base_batch(
            latents,
            vae,
            &device,
            coords.as_slice(),
            batch_blocks.len(),
            points_per_block,
        );
        let data = tensor_to_vec_f32(decoded)
            .map_err(|err| format!("failed to decode flash base grid: {err}"))?;
        write_flash_batch_values(
            &mut grid_values,
            indices.as_slice(),
            data.as_slice(),
            "flash base grid",
        )
        .map_err(std::io::Error::other)?;
    }

    Ok(grid_values)
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
    for batch in coords.chunks(num_chunks) {
        fill_flash_refinement_batch_buffers(batch, size, &mut coord_buf, &mut indices);
        let decoded =
            decode_flash_refinement_batch(latents, vae, &device, coord_buf.as_slice(), batch.len());
        let data = tensor_to_vec_f32(decoded)
            .map_err(|err| format!("failed to decode flash grid values: {err}"))?;
        write_flash_batch_values(
            output,
            indices.as_slice(),
            data.as_slice(),
            "flash refinement",
        )
        .map_err(std::io::Error::other)?;
    }
    Ok(())
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
    let decoded = vae.decode(coords_tensor, latents, None);
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
    let decoded = vae.decode(coords_tensor, latents, None);
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
    use burn::backend::NdArray;

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

    #[test]
    fn flash_base_batch_buffers_cover_all_grid_points() {
        let mini_grid_num = 2usize;
        let grid_size = 4usize;
        let mini_size = 2usize;
        let xs: Vec<f32> = (0..grid_size).map(|x| x as f32).collect();
        let ys: Vec<f32> = (0..grid_size).map(|y| (10 + y) as f32).collect();
        let zs: Vec<f32> = (0..grid_size).map(|z| (20 + z) as f32).collect();

        let blocks = flash_base_blocks(mini_grid_num);
        let mut coords = Vec::new();
        let mut indices = Vec::new();
        let mut seen = vec![false; grid_size * grid_size * grid_size];

        for batch_blocks in blocks.chunks(1) {
            fill_flash_base_batch_buffers(
                batch_blocks,
                mini_size,
                grid_size,
                [&xs, &ys, &zs],
                &mut coords,
                &mut indices,
            );
            assert_eq!(coords.len(), indices.len() * 3);
            for (point_idx, &linear_idx) in indices.iter().enumerate() {
                assert!(!seen[linear_idx], "duplicate linear index {linear_idx}");
                seen[linear_idx] = true;
                let x = linear_idx % grid_size;
                let y = (linear_idx / grid_size) % grid_size;
                let z = linear_idx / (grid_size * grid_size);
                assert_eq!(coords[point_idx * 3], xs[x]);
                assert_eq!(coords[point_idx * 3 + 1], ys[y]);
                assert_eq!(coords[point_idx * 3 + 2], zs[z]);
            }
        }

        assert!(seen.into_iter().all(|flag| flag));
    }

    #[test]
    fn flash_refinement_batch_buffers_preserve_order_and_indices() {
        let coords_src = vec![
            ([1usize, 2usize, 3usize], [0.1f32, 0.2f32, 0.3f32]),
            ([4usize, 5usize, 6usize], [1.1f32, 1.2f32, 1.3f32]),
        ];
        let size = 8usize;
        let mut coords = Vec::new();
        let mut indices = Vec::new();
        fill_flash_refinement_batch_buffers(&coords_src, size, &mut coords, &mut indices);

        assert_eq!(coords, vec![0.1, 0.2, 0.3, 1.1, 1.2, 1.3]);
        assert_eq!(
            indices,
            vec![(3 * size + 2) * size + 1, (6 * size + 5) * size + 4,]
        );
    }

    #[test]
    fn flash_blocks_per_batch_rounds_up_partial_block() {
        let points_per_block = 512usize;
        let num_chunks = points_per_block + 1;
        let blocks = flash_blocks_per_batch::<NdArray<f32>>(num_chunks, points_per_block);
        assert_eq!(blocks, 2);
    }

    #[test]
    fn zero_padded_flash_delta_tail_masks_only_padding() {
        type B = NdArray<f32>;
        let device = <B as Backend>::Device::default();
        let delta = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device);
        let masked = zero_padded_flash_delta_tail(delta, 3);
        let values = masked
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor data to vec");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 0.0]);
    }

    fn build_next_mask_reference(
        curr_mask: &[bool],
        grid_size: usize,
        next_size: usize,
        dilation_iters: usize,
    ) -> Vec<u8> {
        let mut next_mask = vec![0u8; next_size * next_size * next_size];
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let idx = (z * grid_size + y) * grid_size + x;
                    if !curr_mask[idx] {
                        continue;
                    }
                    let nx = x * 2;
                    let ny = y * 2;
                    let nz = z * 2;
                    if nx < next_size && ny < next_size && nz < next_size {
                        let nidx = (nz * next_size + ny) * next_size + nx;
                        next_mask[nidx] = 1;
                    }
                }
            }
        }
        for _ in 0..dilation_iters {
            next_mask = dilate_mask(&next_mask, next_size);
        }
        next_mask
    }

    #[test]
    fn flash_refinement_compact_host_matches_reference_expansion() {
        let grid_size = 5usize;
        let next_size = 9usize;
        let mut curr_mask = vec![false; grid_size * grid_size * grid_size];
        for [x, y, z] in [
            [0usize, 0usize, 0usize],
            [1, 2, 3],
            [4, 4, 4],
            [2, 0, 4],
            [3, 4, 1],
        ] {
            let idx = (z * grid_size + y) * grid_size + x;
            curr_mask[idx] = true;
        }

        for dilation_iters in [1usize, 2usize] {
            let (curr_count, flat_indices, coords) = build_flash_refinement_compact_host(
                curr_mask.as_slice(),
                grid_size,
                next_size,
                dilation_iters,
            )
            .expect("compact refinement");
            assert_eq!(curr_count, 5);

            let ref_mask = build_next_mask_reference(
                curr_mask.as_slice(),
                grid_size,
                next_size,
                dilation_iters,
            );
            let ref_indices: Vec<i32> = ref_mask
                .iter()
                .enumerate()
                .filter_map(|(idx, &active)| if active > 0 { Some(idx as i32) } else { None })
                .collect();
            assert_eq!(flat_indices, ref_indices, "dilation_iters={dilation_iters}");
            assert_eq!(coords.len(), flat_indices.len() * 3);

            let next_plane = next_size * next_size;
            for (point_idx, &linear) in flat_indices.iter().enumerate() {
                let linear = linear as usize;
                let z = linear / next_plane;
                let rem = linear - z * next_plane;
                let y = rem / next_size;
                let x = rem - y * next_size;
                assert_eq!(coords[point_idx * 3], x as i32);
                assert_eq!(coords[point_idx * 3 + 1], y as i32);
                assert_eq!(coords[point_idx * 3 + 2], z as i32);
            }
        }
    }

    #[test]
    fn flash_refinement_compact_host_empty_mask_returns_empty_outputs() {
        let grid_size = 4usize;
        let next_size = 7usize;
        let curr_mask = vec![false; grid_size * grid_size * grid_size];
        let (curr_count, flat_indices, coords) =
            build_flash_refinement_compact_host(curr_mask.as_slice(), grid_size, next_size, 2)
                .expect("compact refinement");
        assert_eq!(curr_count, 0);
        assert!(flat_indices.is_empty());
        assert!(coords.is_empty());
    }
}

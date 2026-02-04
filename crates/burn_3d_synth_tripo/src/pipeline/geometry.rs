use burn::prelude::*;

use crate::model::triposg::vae::TripoSGVae;
use crate::pipeline::mesh::DenseGrid;

#[derive(Debug, Clone)]
pub struct HierarchicalExtractConfig {
    pub bounds: [f32; 6],
    pub dense_octree_depth: usize,
    pub hierarchical_octree_depth: usize,
    pub chunk_size: usize,
    pub band_threshold: f32,
}

impl HierarchicalExtractConfig {
    pub fn new(bounds: [f32; 6], dense_octree_depth: usize, hierarchical_octree_depth: usize) -> Self {
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
    let mut indices = Vec::with_capacity(chunk_size);
    let mut idx = 0usize;

    for &zv in zs.iter() {
        for &yv in ys.iter() {
            for &xv in xs.iter() {
                coords.push(xv);
                coords.push(yv);
                coords.push(zv);
                indices.push(idx);
                idx += 1;

                if indices.len() >= chunk_size {
                    write_decoded(latents, vae, &coords, &indices, &device, &mut values)?;
                    coords.clear();
                    indices.clear();
                }
            }
        }
    }

    if !indices.is_empty() {
        write_decoded(latents, vae, &coords, &indices, &device, &mut values)?;
    }

    Ok(values)
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
    let coords_tensor = Tensor::<B, 1>::from_floats(coords, device)
        .reshape([1, count as i32, 3]);
    let decoded = vae.decode(coords_tensor, latents.clone(), None);
    let data = decoded
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to decode grid values")?;
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

fn find_candidates_band(
    values: &[f32],
    size: usize,
    band_threshold: f32,
) -> Vec<[usize; 3]> {
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

fn expand_edge_region(
    coords: &[[usize; 3]],
    low_size: usize,
    high_size: usize,
) -> Vec<[usize; 3]> {
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

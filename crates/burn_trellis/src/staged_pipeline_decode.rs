use super::*;
use crate::time::Instant;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault};

#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{DefaultWgpuBackend, dense_trilinear_sample_attrs_wgpu};
#[cfg(feature = "runtime-model-wgpu")]
use burn_wgpu::WgpuDevice;
use rustc_hash::FxHasher;

// Internal helpers for TRELLIS staged decode, mesh extraction, and PBR baking.
// Kept in a separate module so `staged_pipeline.rs` stays focused on stage orchestration.

#[derive(Debug, Clone)]
pub(super) struct UvRasterDomain {
    pub(super) output_vertices: Vec<[f32; 3]>,
    pub(super) output_faces: Vec<[u32; 3]>,
    pub(super) output_uvs: Vec<[f32; 2]>,
    pub(super) raster_vertices: Vec<[f32; 3]>,
    pub(super) raster_uvs: Vec<[f32; 2]>,
    pub(super) raster_faces: Vec<[u32; 3]>,
}

#[derive(Debug, Clone)]
pub(super) struct PbrBakeMesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
}

type VoxelAttrFastHasher = BuildHasherDefault<FxHasher>;
type VoxelAttrMap = HashMap<u64, [f32; 6], VoxelAttrFastHasher>;
type FastHashMap<K, V> = HashMap<K, V, VoxelAttrFastHasher>;
// Keep dense lookup bounded to avoid excessive host memory use when sparse
// coords span large volumes; large cases stay on sparse hash lookup.
const DENSE_VOXEL_LOOKUP_MAX_CELLS: usize = 2_500_000;
#[cfg(feature = "runtime-model-wgpu")]
const DENSE_VOXEL_WGPU_SAMPLE_MIN_POSITIONS: usize = 2_048;
#[cfg(feature = "runtime-model-wgpu")]
const DENSE_VOXEL_WGPU_SAMPLE_BATCH: usize = 65_536;
#[cfg(feature = "runtime-model-wgpu")]
const DENSE_VOXEL_WGPU_MAX_CANDIDATES_PER_TEXEL: usize = 8;
const PBR_PROJECTION_BVH_LEAF_TRIANGLES: usize = 16;
const PBR_UV_ATLAS_TARGET_OCCUPANCY: f32 = 0.88;

#[derive(Debug, Clone, Copy)]
pub(super) struct PbrProjectionSource<'a> {
    pub vertices: &'a [[f32; 3]],
    pub faces: &'a [[u32; 3]],
}

#[derive(Debug, Clone, Copy)]
struct ProjectionAabb {
    min: [f32; 3],
    max: [f32; 3],
}

#[derive(Debug, Clone, Copy)]
struct ProjectionBvhNode {
    bounds: ProjectionAabb,
    start: usize,
    len: usize,
    left: usize,
    right: usize,
}

pub(super) struct ProjectionBvh<'a> {
    vertices: &'a [[f32; 3]],
    faces: &'a [[u32; 3]],
    tri_indices: Vec<usize>,
    nodes: Vec<ProjectionBvhNode>,
    root: usize,
}

fn pbr_stage_log(message: impl AsRef<str>) {
    eprintln!("[{}] {}", stage_log_timestamp(), message.as_ref());
}

fn fast_hash_map_with_capacity<K, V>(capacity: usize) -> FastHashMap<K, V> {
    HashMap::with_capacity_and_hasher(capacity, VoxelAttrFastHasher::default())
}

pub(super) enum VoxelAttrLookup {
    Dense {
        spatial: [u32; 3],
        occupancy: Vec<u8>,
        attrs: Vec<[f32; 6]>,
    },
    Sparse {
        spatial: [u32; 3],
        map: VoxelAttrMap,
    },
}

#[cfg(feature = "runtime-model-wgpu")]
struct DenseVoxelWgpuSampler {
    device: WgpuDevice,
    occupancy_t: Tensor<DefaultWgpuBackend, 1, Int>,
    attrs_t: Tensor<DefaultWgpuBackend, 2>,
    spatial: [usize; 3],
}

#[cfg(feature = "runtime-model-wgpu")]
impl DenseVoxelWgpuSampler {
    fn new(occupancy: &[u8], attrs: &[[f32; 6]], spatial: [u32; 3]) -> Result<Self, String> {
        if occupancy.is_empty() || attrs.is_empty() {
            return Err("decode pbr sample requires non-empty voxel map".to_string());
        }
        if occupancy.len() != attrs.len() {
            return Err(format!(
                "decode pbr dense lookup mismatch: occupancy={} attrs={}",
                occupancy.len(),
                attrs.len()
            ));
        }
        let spatial_cells = (spatial[0] as usize)
            .checked_mul(spatial[1] as usize)
            .and_then(|value| value.checked_mul(spatial[2] as usize))
            .ok_or_else(|| {
                format!(
                    "decode pbr dense lookup volume overflow: spatial=[{},{},{}]",
                    spatial[0], spatial[1], spatial[2]
                )
            })?;
        if spatial_cells != occupancy.len() {
            return Err(format!(
                "decode pbr dense lookup length mismatch: expected_cells={} occupancy={} attrs={}",
                spatial_cells,
                occupancy.len(),
                attrs.len()
            ));
        }

        let spatial_usize = [
            usize::try_from(spatial[0])
                .map_err(|_| format!("decode pbr spatial x={} exceeds usize range", spatial[0]))?,
            usize::try_from(spatial[1])
                .map_err(|_| format!("decode pbr spatial y={} exceeds usize range", spatial[1]))?,
            usize::try_from(spatial[2])
                .map_err(|_| format!("decode pbr spatial z={} exceeds usize range", spatial[2]))?,
        ];

        let device = WgpuDevice::default();
        let mut attrs_flat = Vec::with_capacity(attrs.len().saturating_mul(6));
        for row in attrs {
            attrs_flat.extend_from_slice(row);
        }
        let occupancy_i32 = occupancy
            .iter()
            .map(|value| i32::from(*value > 0))
            .collect::<Vec<_>>();
        let attrs_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
            TensorData::new(attrs_flat, [attrs.len(), 6]),
            &device,
        );
        let occupancy_t = Tensor::<DefaultWgpuBackend, 1, Int>::from_data(
            TensorData::new(occupancy_i32, [occupancy.len()]),
            &device,
        );

        Ok(Self {
            device,
            occupancy_t,
            attrs_t,
            spatial: spatial_usize,
        })
    }
}

impl<'a> ProjectionBvh<'a> {
    fn build(source: PbrProjectionSource<'a>) -> Result<Self, String> {
        if source.vertices.is_empty() || source.faces.is_empty() {
            return Err("decode pbr projection requires non-empty source mesh".to_string());
        }
        let mut face_centroids = vec![[0.0f32; 3]; source.faces.len()];
        let mut tri_indices = Vec::with_capacity(source.faces.len());
        for (idx, face) in source.faces.iter().copied().enumerate() {
            if face
                .iter()
                .all(|index| (*index as usize) < source.vertices.len())
            {
                face_centroids[idx] = triangle_centroid(source.vertices, face);
                tri_indices.push(idx);
            }
        }
        if tri_indices.is_empty() {
            return Err("decode pbr projection source has no valid triangles".to_string());
        }

        let tri_count = tri_indices.len();
        let mut nodes = Vec::with_capacity(tri_count.saturating_mul(2).max(1));
        let root = build_projection_bvh_node(
            source.vertices,
            source.faces,
            face_centroids.as_slice(),
            tri_indices.as_mut_slice(),
            &mut nodes,
            0,
            tri_count,
        )?;
        Ok(Self {
            vertices: source.vertices,
            faces: source.faces,
            tri_indices,
            nodes,
            root,
        })
    }

    fn closest_point(&self, point: [f32; 3]) -> [f32; 3] {
        self.closest_point_and_distance2(point).0
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn closest_distance2(&self, point: [f32; 3]) -> f32 {
        self.closest_point_and_distance2(point).1
    }

    fn closest_point_and_distance2(&self, point: [f32; 3]) -> ([f32; 3], f32) {
        let mut best_dist2 = f32::INFINITY;
        let mut best_point = point;
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(node_idx) = stack.pop() {
            let node = self.nodes[node_idx];
            if node.bounds.distance2(point) > best_dist2 {
                continue;
            }
            if node.len > 0 {
                for tri_slot in node.start..node.start + node.len {
                    let face = self.faces[self.tri_indices[tri_slot]];
                    let a = self.vertices[face[0] as usize];
                    let b = self.vertices[face[1] as usize];
                    let c = self.vertices[face[2] as usize];
                    let candidate = closest_point_on_triangle(point, a, b, c);
                    let dist2 = vec3_len2(vec3_sub(point, candidate));
                    if dist2 < best_dist2 {
                        best_dist2 = dist2;
                        best_point = candidate;
                    }
                }
                continue;
            }

            let left_dist2 = self.nodes[node.left].bounds.distance2(point);
            let right_dist2 = self.nodes[node.right].bounds.distance2(point);
            if left_dist2 <= right_dist2 {
                if right_dist2 <= best_dist2 {
                    stack.push(node.right);
                }
                if left_dist2 <= best_dist2 {
                    stack.push(node.left);
                }
            } else {
                if left_dist2 <= best_dist2 {
                    stack.push(node.left);
                }
                if right_dist2 <= best_dist2 {
                    stack.push(node.right);
                }
            }
        }
        (best_point, best_dist2)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn build_projection_bvh_for_pbr(
    source: PbrProjectionSource<'_>,
) -> Result<ProjectionBvh<'_>, String> {
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr remesh_bvh begin (vertices={} faces={})",
        source.vertices.len(),
        source.faces.len()
    ));
    let bvh_start = Instant::now();
    let bvh = ProjectionBvh::build(source)?;
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr remesh_bvh complete ({:.2} ms)",
        bvh_start.elapsed().as_secs_f64() * 1000.0
    ));
    Ok(bvh)
}

#[cfg(not(target_arch = "wasm32"))]
pub(super) fn remesh_narrow_band_simple_dc_with_projection_bvh(
    bvh: &ProjectionBvh<'_>,
    final_resolution: u32,
    band: f32,
) -> Result<(Vec<[f32; 3]>, Vec<[u32; 3]>), String> {
    let resolution = final_resolution.max(1);
    if resolution < 2 {
        return Err(format!(
            "native pbr remesh requires resolution >= 2, got {resolution}"
        ));
    }

    let domain_scale = (resolution as f32 + 3.0 * band) / resolution as f32;
    let eps = band * domain_scale / resolution as f32;
    let mut base_resolution = resolution;
    while base_resolution > 32 {
        if base_resolution % 2 != 0 {
            return Err(format!(
                "native pbr remesh resolution must be divisible by two down to <=32, got {resolution}"
            ));
        }
        base_resolution /= 2;
    }

    let mut coords = Vec::<[i32; 3]>::with_capacity(
        (base_resolution as usize)
            .saturating_mul(base_resolution as usize)
            .saturating_mul(base_resolution as usize),
    );
    for x in 0..base_resolution as i32 {
        for y in 0..base_resolution as i32 {
            for z in 0..base_resolution as i32 {
                coords.push([x, y, z]);
            }
        }
    }

    let mut current_resolution = base_resolution;
    let refine_start = Instant::now();
    loop {
        let cell_size = domain_scale / current_resolution as f32;
        let threshold = 0.87 * cell_size;
        coords.retain(|coord| {
            let point = remesh_cell_center(*coord, current_resolution, domain_scale);
            let distance = bvh.closest_distance2(point).sqrt();
            (distance - eps).abs() < threshold
        });
        pbr_stage_log(format!(
            "burn_trellis: decode.pbr remesh refine level complete (resolution={} active_voxels={})",
            current_resolution,
            coords.len()
        ));
        if current_resolution >= resolution {
            break;
        }

        current_resolution *= 2;
        let mut children = Vec::with_capacity(coords.len().saturating_mul(8));
        for coord in coords {
            for dx in 0..=1 {
                for dy in 0..=1 {
                    for dz in 0..=1 {
                        children.push([coord[0] * 2 + dx, coord[1] * 2 + dy, coord[2] * 2 + dz]);
                    }
                }
            }
        }
        coords = children;
    }
    if coords.is_empty() {
        return Err("native pbr remesh produced no active voxels".to_string());
    }
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr remesh refine complete ({:.2} ms, active_voxels={})",
        refine_start.elapsed().as_secs_f64() * 1000.0,
        coords.len()
    ));

    let dc_start = Instant::now();
    let mut grid_vertex_map =
        fast_hash_map_with_capacity::<(i32, i32, i32), usize>(coords.len().saturating_mul(4));
    let mut grid_vertices = Vec::<[i32; 3]>::with_capacity(coords.len().saturating_mul(4));
    for coord in &coords {
        for corner in REMESH_CORNERS {
            let key = (
                coord[0] + corner[0],
                coord[1] + corner[1],
                coord[2] + corner[2],
            );
            grid_vertex_map.entry(key).or_insert_with(|| {
                let idx = grid_vertices.len();
                grid_vertices.push([key.0, key.1, key.2]);
                idx
            });
        }
    }
    let mut grid_values = Vec::with_capacity(grid_vertices.len());
    for vertex in &grid_vertices {
        let point = remesh_grid_point(*vertex, resolution, domain_scale);
        grid_values.push(bvh.closest_distance2(point).sqrt() - eps);
    }

    let mut voxel_map = fast_hash_map_with_capacity::<(i32, i32, i32), usize>(coords.len());
    for (idx, coord) in coords.iter().copied().enumerate() {
        voxel_map.insert((coord[0], coord[1], coord[2]), idx);
    }

    let mut dual_vertices_grid = Vec::<[f32; 3]>::with_capacity(coords.len());
    let mut intersected = Vec::<[i32; 3]>::with_capacity(coords.len());
    for coord in &coords {
        let (dual, edge_flags) =
            simple_dual_contour_voxel(*coord, &grid_vertex_map, grid_values.as_slice())?;
        dual_vertices_grid.push(dual);
        intersected.push(edge_flags);
    }

    let mut used = vec![false; coords.len()];
    let mut remap = vec![u32::MAX; coords.len()];
    let mut faces_out = Vec::<[u32; 3]>::new();
    for (voxel_idx, coord) in coords.iter().copied().enumerate() {
        for axis in 0..3 {
            let dir = intersected[voxel_idx][axis];
            if dir == 0 {
                continue;
            }
            let mut quad = [0usize; 4];
            let mut valid = true;
            for (slot, offset) in REMESH_EDGE_NEIGHBOR_VOXEL_OFFSET[axis].iter().enumerate() {
                let key = (
                    coord[0] + offset[0],
                    coord[1] + offset[1],
                    coord[2] + offset[2],
                );
                let Some(&neighbor_idx) = voxel_map.get(&key) else {
                    valid = false;
                    break;
                };
                quad[slot] = neighbor_idx;
            }
            if !valid {
                continue;
            }
            let triangles_0 = remesh_quad_triangles(quad, dir, 0);
            let triangles_1 = remesh_quad_triangles(quad, dir, 1);
            let align0 = remesh_split_alignment(dual_vertices_grid.as_slice(), triangles_0);
            let align1 = remesh_split_alignment(dual_vertices_grid.as_slice(), triangles_1);
            let selected = if align0 > align1 {
                triangles_0
            } else {
                triangles_1
            };
            for face in selected {
                for idx in face {
                    used[idx] = true;
                }
                faces_out.push([face[0] as u32, face[1] as u32, face[2] as u32]);
            }
        }
    }
    if faces_out.is_empty() {
        return Err("native pbr remesh produced no faces".to_string());
    }

    let mut vertices_out = Vec::with_capacity(used.iter().filter(|value| **value).count());
    for (idx, value) in used.into_iter().enumerate() {
        if !value {
            continue;
        }
        remap[idx] = vertices_out.len() as u32;
        vertices_out.push(remesh_grid_to_world(
            dual_vertices_grid[idx],
            resolution,
            domain_scale,
        ));
    }
    for face in &mut faces_out {
        face[0] = remap[face[0] as usize];
        face[1] = remap[face[1] as usize];
        face[2] = remap[face[2] as usize];
    }
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr remesh dc complete ({:.2} ms, grid_vertices={} vertices={} faces={})",
        dc_start.elapsed().as_secs_f64() * 1000.0,
        grid_vertices.len(),
        vertices_out.len(),
        faces_out.len()
    ));
    Ok((vertices_out, faces_out))
}

#[cfg(not(target_arch = "wasm32"))]
const REMESH_CORNERS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

#[cfg(not(target_arch = "wasm32"))]
const REMESH_EDGE_NEIGHBOR_VOXEL_OFFSET: [[[i32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];

#[cfg(not(target_arch = "wasm32"))]
fn remesh_cell_center(coord: [i32; 3], resolution: u32, domain_scale: f32) -> [f32; 3] {
    [
        ((coord[0] as f32 + 0.5) / resolution as f32 - 0.5) * domain_scale,
        ((coord[1] as f32 + 0.5) / resolution as f32 - 0.5) * domain_scale,
        ((coord[2] as f32 + 0.5) / resolution as f32 - 0.5) * domain_scale,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_grid_point(coord: [i32; 3], resolution: u32, domain_scale: f32) -> [f32; 3] {
    [
        (coord[0] as f32 / resolution as f32 - 0.5) * domain_scale,
        (coord[1] as f32 / resolution as f32 - 0.5) * domain_scale,
        (coord[2] as f32 / resolution as f32 - 0.5) * domain_scale,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_grid_to_world(coord: [f32; 3], resolution: u32, domain_scale: f32) -> [f32; 3] {
    [
        (coord[0] / resolution as f32 - 0.5) * domain_scale,
        (coord[1] / resolution as f32 - 0.5) * domain_scale,
        (coord[2] / resolution as f32 - 0.5) * domain_scale,
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn simple_dual_contour_voxel(
    coord: [i32; 3],
    grid_vertex_map: &FastHashMap<(i32, i32, i32), usize>,
    grid_values: &[f32],
) -> Result<([f32; 3], [i32; 3]), String> {
    let value = |x: i32, y: i32, z: i32| -> Result<f32, String> {
        let idx = grid_vertex_map
            .get(&(x, y, z))
            .copied()
            .ok_or_else(|| format!("native pbr remesh missing grid vertex ({x},{y},{z})"))?;
        grid_values
            .get(idx)
            .copied()
            .ok_or_else(|| format!("native pbr remesh grid value index out of range: {idx}"))
    };

    let [vx, vy, vz] = coord;
    let mut intersection_sum = [0.0f32; 3];
    let mut intersection_count = 0usize;
    let mut intersected = [0i32; 3];

    for u in 0..=1 {
        for v in 0..=1 {
            let val1 = value(vx, vy + u, vz + v)?;
            let val2 = value(vx + 1, vy + u, vz + v)?;
            if remesh_edge_crosses(val1, val2) {
                let t = remesh_interp_t(val1, val2);
                intersection_sum[0] += vx as f32 + t;
                intersection_sum[1] += (vy + u) as f32;
                intersection_sum[2] += (vz + v) as f32;
                intersection_count += 1;
            }
            if u == 1 && v == 1 {
                intersected[0] = remesh_intersection_dir(val1, val2);
            }
        }
    }
    for u in 0..=1 {
        for v in 0..=1 {
            let val1 = value(vx + u, vy, vz + v)?;
            let val2 = value(vx + u, vy + 1, vz + v)?;
            if remesh_edge_crosses(val1, val2) {
                let t = remesh_interp_t(val1, val2);
                intersection_sum[0] += (vx + u) as f32;
                intersection_sum[1] += vy as f32 + t;
                intersection_sum[2] += (vz + v) as f32;
                intersection_count += 1;
            }
            if u == 1 && v == 1 {
                intersected[1] = remesh_intersection_dir(val1, val2);
            }
        }
    }
    for u in 0..=1 {
        for v in 0..=1 {
            let val1 = value(vx + u, vy + v, vz)?;
            let val2 = value(vx + u, vy + v, vz + 1)?;
            if remesh_edge_crosses(val1, val2) {
                let t = remesh_interp_t(val1, val2);
                intersection_sum[0] += (vx + u) as f32;
                intersection_sum[1] += (vy + v) as f32;
                intersection_sum[2] += vz as f32 + t;
                intersection_count += 1;
            }
            if u == 1 && v == 1 {
                intersected[2] = remesh_intersection_dir(val1, val2);
            }
        }
    }

    let dual = if intersection_count > 0 {
        [
            intersection_sum[0] / intersection_count as f32,
            intersection_sum[1] / intersection_count as f32,
            intersection_sum[2] / intersection_count as f32,
        ]
    } else {
        [vx as f32 + 0.5, vy as f32 + 0.5, vz as f32 + 0.5]
    };
    Ok((dual, intersected))
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_edge_crosses(a: f32, b: f32) -> bool {
    (a < 0.0 && b >= 0.0) || (a >= 0.0 && b < 0.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_interp_t(a: f32, b: f32) -> f32 {
    let denom = b - a;
    if denom.abs() <= 1.0e-20 {
        0.5
    } else {
        (-a / denom).clamp(0.0, 1.0)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_intersection_dir(a: f32, b: f32) -> i32 {
    if a < 0.0 && b >= 0.0 {
        1
    } else if a >= 0.0 && b < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_quad_triangles(quad: [usize; 4], dir: i32, split: usize) -> [[usize; 3]; 2] {
    let pattern = match (split, dir == 1) {
        (0, true) => [0, 2, 1, 0, 3, 2],
        (0, false) => [0, 1, 2, 0, 2, 3],
        (_, true) => [0, 3, 1, 3, 2, 1],
        (_, false) => [0, 1, 3, 3, 1, 2],
    };
    [
        [quad[pattern[0]], quad[pattern[1]], quad[pattern[2]]],
        [quad[pattern[3]], quad[pattern[4]], quad[pattern[5]]],
    ]
}

#[cfg(not(target_arch = "wasm32"))]
fn remesh_split_alignment(vertices: &[[f32; 3]], triangles: [[usize; 3]; 2]) -> f32 {
    let normal0 = triangle_normal_usize(vertices, triangles[0]);
    let normal1 = triangle_normal_usize(vertices, triangles[1]);
    vec3_dot(normal0, normal1).abs()
}

#[cfg(not(target_arch = "wasm32"))]
fn triangle_normal_usize(vertices: &[[f32; 3]], face: [usize; 3]) -> [f32; 3] {
    let a = vertices[face[0]];
    let b = vertices[face[1]];
    let c = vertices[face[2]];
    vec3_cross(vec3_sub(b, a), vec3_sub(c, a))
}

impl ProjectionAabb {
    fn empty() -> Self {
        Self {
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
        }
    }

    fn include_point(&mut self, point: [f32; 3]) {
        for (axis, value) in point.iter().copied().enumerate() {
            self.min[axis] = self.min[axis].min(value);
            self.max[axis] = self.max[axis].max(value);
        }
    }

    fn include_triangle(&mut self, vertices: &[[f32; 3]], face: [u32; 3]) {
        self.include_point(vertices[face[0] as usize]);
        self.include_point(vertices[face[1] as usize]);
        self.include_point(vertices[face[2] as usize]);
    }

    fn extent(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    fn distance2(self, point: [f32; 3]) -> f32 {
        let mut out = 0.0f32;
        for (axis, value) in point.iter().copied().enumerate() {
            let delta = if value < self.min[axis] {
                self.min[axis] - value
            } else if value > self.max[axis] {
                value - self.max[axis]
            } else {
                0.0
            };
            out += delta * delta;
        }
        out
    }
}

fn build_projection_bvh_node(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    face_centroids: &[[f32; 3]],
    tri_indices: &mut [usize],
    nodes: &mut Vec<ProjectionBvhNode>,
    start: usize,
    end: usize,
) -> Result<usize, String> {
    if start >= end {
        return Err("decode pbr projection BVH received empty node range".to_string());
    }

    let mut bounds = ProjectionAabb::empty();
    let mut centroid_bounds = ProjectionAabb::empty();
    for slot in start..end {
        let face = faces[tri_indices[slot]];
        bounds.include_triangle(vertices, face);
        centroid_bounds.include_point(face_centroids[tri_indices[slot]]);
    }

    let len = end - start;
    if len <= PBR_PROJECTION_BVH_LEAF_TRIANGLES {
        let node_idx = nodes.len();
        nodes.push(ProjectionBvhNode {
            bounds,
            start,
            len,
            left: usize::MAX,
            right: usize::MAX,
        });
        return Ok(node_idx);
    }

    let extent = centroid_bounds.extent();
    let split_axis = if extent[0] >= extent[1] && extent[0] >= extent[2] {
        0
    } else if extent[1] >= extent[2] {
        1
    } else {
        2
    };
    tri_indices[start..end].sort_unstable_by(|a, b| {
        let ca = face_centroids[*a][split_axis];
        let cb = face_centroids[*b][split_axis];
        ca.total_cmp(&cb).then_with(|| a.cmp(b))
    });
    let mid = start + len / 2;
    if mid == start || mid == end {
        let node_idx = nodes.len();
        nodes.push(ProjectionBvhNode {
            bounds,
            start,
            len,
            left: usize::MAX,
            right: usize::MAX,
        });
        return Ok(node_idx);
    }

    let left = build_projection_bvh_node(
        vertices,
        faces,
        face_centroids,
        tri_indices,
        nodes,
        start,
        mid,
    )?;
    let right = build_projection_bvh_node(
        vertices,
        faces,
        face_centroids,
        tri_indices,
        nodes,
        mid,
        end,
    )?;
    let node_idx = nodes.len();
    nodes.push(ProjectionBvhNode {
        bounds,
        start: 0,
        len: 0,
        left,
        right,
    });
    Ok(node_idx)
}

fn triangle_centroid(vertices: &[[f32; 3]], face: [u32; 3]) -> [f32; 3] {
    let a = vertices[face[0] as usize];
    let b = vertices[face[1] as usize];
    let c = vertices[face[2] as usize];
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ]
}

fn closest_point_on_triangle(p: [f32; 3], a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = vec3_sub(b, a);
    let ac = vec3_sub(c, a);
    let ap = vec3_sub(p, a);
    let d1 = vec3_dot(ab, ap);
    let d2 = vec3_dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = vec3_sub(p, b);
    let d3 = vec3_dot(ab, bp);
    let d4 = vec3_dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3).max(1.0e-20);
        return vec3_add(a, vec3_mul(ab, v));
    }

    let cp = vec3_sub(p, c);
    let d5 = vec3_dot(ab, cp);
    let d6 = vec3_dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6).max(1.0e-20);
        return vec3_add(a, vec3_mul(ac, w));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let bc = vec3_sub(c, b);
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)).max(1.0e-20);
        return vec3_add(b, vec3_mul(bc, w));
    }

    let denom = (va + vb + vc).max(1.0e-20);
    let v = vb / denom;
    let w = vc / denom;
    vec3_add(a, vec3_add(vec3_mul(ab, v), vec3_mul(ac, w)))
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn vec3_add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn vec3_mul(a: [f32; 3], scale: f32) -> [f32; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}

#[cfg(not(target_arch = "wasm32"))]
fn vec3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn vec3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn vec3_len2(a: [f32; 3]) -> f32 {
    vec3_dot(a, a)
}

#[cfg(feature = "runtime-model")]
pub(super) fn runtime_subdivision_to_sample(
    sub: &SparseSubdivisionLogits,
) -> Result<DecodeShapeSubSample, String> {
    let coords = sub.coords_host("decode runtime subdivision coord materialization")?;
    let logits = sub.logits_host("decode runtime subdivision logits materialization")?;
    if logits.len() != coords.len().saturating_mul(8) {
        return Err(format!(
            "decode runtime subdivision tensor mismatch: coords_rows={} logits={}",
            coords.len(),
            logits.len()
        ));
    }
    let mut feats = Vec::with_capacity(coords.len());
    for row_idx in 0..coords.len() {
        let mut row = [0.0f32; 8];
        let base = row_idx * 8;
        row.copy_from_slice(&logits[base..base + 8]);
        feats.push(row);
    }
    let spatial_shape = spatial_shape_from_sparse_coords(coords.as_slice());
    Ok(DecodeShapeSubSample {
        coords,
        feats,
        spatial_shape,
    })
}

#[cfg(feature = "runtime-model")]
pub(super) fn spatial_shape_from_sparse_coords(coords: &[[u32; 4]]) -> [u32; 3] {
    if coords.is_empty() {
        return [1, 1, 1];
    }
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut max_z = 0u32;
    for coord in coords {
        max_x = max_x.max(coord[1]);
        max_y = max_y.max(coord[2]);
        max_z = max_z.max(coord[3]);
    }
    [
        max_x.saturating_add(1),
        max_y.saturating_add(1),
        max_z.saturating_add(1),
    ]
}

pub(super) fn flexible_dual_grid_to_mesh(
    coords: &[[u32; 4]],
    dual_vertices: &[[f32; 3]],
    intersected_flag: &[[bool; 3]],
    split_weight: Option<&[f32]>,
    grid_size: [u32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    if coords.is_empty()
        || dual_vertices.len() != coords.len()
        || intersected_flag.len() != coords.len()
        || split_weight.is_some_and(|w| w.len() != coords.len())
    {
        return (Vec::new(), Vec::new());
    }

    // TRELLIS2 flexible-dual-grid edge neighborhoods:
    // x-axis, y-axis, z-axis (4 voxels per quad candidate).
    const EDGE_NEIGHBOR_VOXEL_OFFSET: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];

    let mut coord_to_index = HashMap::with_capacity(coords.len() * 2);
    for (idx, coord) in coords.iter().enumerate() {
        coord_to_index.insert(pack_coord(coord[1], coord[2], coord[3]), idx as u32);
    }

    let mut quad_indices = Vec::<[u32; 4]>::new();
    for (idx, coord) in coords.iter().enumerate() {
        let base = [coord[1] as i32, coord[2] as i32, coord[3] as i32];
        for axis in 0..3 {
            if !intersected_flag[idx][axis] {
                continue;
            }
            let mut quad = [0u32; 4];
            let mut valid = true;
            for k in 0..4 {
                let offset = EDGE_NEIGHBOR_VOXEL_OFFSET[axis][k];
                let nx = base[0] + offset[0];
                let ny = base[1] + offset[1];
                let nz = base[2] + offset[2];
                if nx < 0 || ny < 0 || nz < 0 {
                    valid = false;
                    break;
                }
                let Some(&neighbor_idx) =
                    coord_to_index.get(&pack_coord(nx as u32, ny as u32, nz as u32))
                else {
                    valid = false;
                    break;
                };
                quad[k] = neighbor_idx;
            }
            if valid {
                quad_indices.push(quad);
            }
        }
    }

    if quad_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let voxel_size = [
        (aabb_max[0] - aabb_min[0]) / grid_size[0].max(1) as f32,
        (aabb_max[1] - aabb_min[1]) / grid_size[1].max(1) as f32,
        (aabb_max[2] - aabb_min[2]) / grid_size[2].max(1) as f32,
    ];
    let mut vertices = Vec::with_capacity(coords.len());
    for (coord, dual) in coords.iter().zip(dual_vertices.iter()) {
        vertices.push([
            (coord[1] as f32 + dual[0]) * voxel_size[0] + aabb_min[0],
            (coord[2] as f32 + dual[1]) * voxel_size[1] + aabb_min[1],
            (coord[3] as f32 + dual[2]) * voxel_size[2] + aabb_min[2],
        ]);
    }

    let mut faces = Vec::with_capacity(quad_indices.len() * 2);
    for quad in quad_indices {
        let use_split_1 = if let Some(weights) = split_weight {
            let w02 = weights[quad[0] as usize] * weights[quad[2] as usize];
            let w13 = weights[quad[1] as usize] * weights[quad[3] as usize];
            w02 > w13
        } else {
            let split1 = quad_to_triangles_split1(quad);
            let split2 = quad_to_triangles_split2(quad);
            triangle_alignment(vertices.as_slice(), split1).abs()
                > triangle_alignment(vertices.as_slice(), split2).abs()
        };
        let tris = if use_split_1 {
            quad_to_triangles_split1(quad)
        } else {
            quad_to_triangles_split2(quad)
        };
        faces.push(tris[0]);
        faces.push(tris[1]);
    }

    (vertices, faces)
}

pub(super) fn quad_to_triangles_split1(quad: [u32; 4]) -> [[u32; 3]; 2] {
    [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
}

pub(super) fn quad_to_triangles_split2(quad: [u32; 4]) -> [[u32; 3]; 2] {
    [[quad[0], quad[1], quad[3]], [quad[3], quad[1], quad[2]]]
}

pub(super) fn triangle_alignment(vertices: &[[f32; 3]], tris: [[u32; 3]; 2]) -> f32 {
    let n0 = triangle_normal(vertices, tris[0]);
    let n1 = triangle_normal(vertices, tris[1]);
    dot3(n0, n1)
}

pub(super) fn triangle_normal(vertices: &[[f32; 3]], tri: [u32; 3]) -> [f32; 3] {
    let a = vertices[tri[0] as usize];
    let b = vertices[tri[1] as usize];
    let c = vertices[tri[2] as usize];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    cross3(ab, ac)
}

pub(super) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(super) fn pack_coord(x: u32, y: u32, z: u32) -> u64 {
    ((x as u64) << 42) | ((y as u64) << 21) | z as u64
}

pub(super) fn summarize_material(
    voxel_attrs: &[[f32; 6]],
    pbr_textures: Option<&MeshPbrTextures>,
) -> Option<MeshMaterial> {
    if let Some(textures) = pbr_textures {
        let base = &textures.base_color.rgba8;
        let mr = &textures.metallic_roughness.rgba8;
        if base.len() >= 4 && mr.len() >= 4 {
            let texels = (base.len() / 4).max(1);
            let mut accum = [0.0f32; 6];
            for idx in 0..texels {
                let off = idx * 4;
                accum[0] += base[off] as f32 / 255.0;
                accum[1] += base[off + 1] as f32 / 255.0;
                accum[2] += base[off + 2] as f32 / 255.0;
                accum[5] += base[off + 3] as f32 / 255.0;
                accum[3] += mr[off + 2] as f32 / 255.0;
                accum[4] += mr[off + 1] as f32 / 255.0;
            }
            let inv = 1.0 / texels as f32;
            return Some(MeshMaterial {
                base_color: [
                    (accum[0] * inv).clamp(0.0, 1.0),
                    (accum[1] * inv).clamp(0.0, 1.0),
                    (accum[2] * inv).clamp(0.0, 1.0),
                ],
                metallic: (accum[3] * inv).clamp(0.0, 1.0),
                roughness: (accum[4] * inv).clamp(0.0, 1.0),
                alpha: (accum[5] * inv).clamp(0.0, 1.0),
            });
        }
    }
    if voxel_attrs.is_empty() {
        return None;
    }
    let mut accum = [0.0f32; 6];
    for attrs in voxel_attrs {
        for idx in 0..6 {
            accum[idx] += attrs[idx];
        }
    }
    let inv = 1.0 / voxel_attrs.len() as f32;
    Some(MeshMaterial {
        base_color: [
            (accum[0] * inv).clamp(0.0, 1.0),
            (accum[1] * inv).clamp(0.0, 1.0),
            (accum[2] * inv).clamp(0.0, 1.0),
        ],
        metallic: (accum[3] * inv).clamp(0.0, 1.0),
        roughness: (accum[4] * inv).clamp(0.0, 1.0),
        alpha: (accum[5] * inv).clamp(0.0, 1.0),
    })
}

const DEFAULT_RUNTIME_PBR_TEXTURE_SIZE: usize = 256;

pub(super) fn runtime_pbr_texture_size(requested: Option<usize>) -> usize {
    requested
        .filter(|size| *size > 0)
        .unwrap_or(DEFAULT_RUNTIME_PBR_TEXTURE_SIZE)
}

#[allow(clippy::type_complexity)]
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn bake_pbr_from_voxels(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
) -> Result<(Vec<[f32; 2]>, Option<MeshPbrTextures>, PbrBakeDebug), String> {
    let (mesh, textures, debug) = bake_pbr_from_voxels_with_options(
        vertices,
        faces,
        None,
        voxel_coords,
        voxel_attrs,
        fallback_spatial_resolution,
        None,
        true,
        false,
    )?;
    Ok((
        mesh.uvs,
        textures,
        debug.unwrap_or_else(empty_pbr_bake_debug),
    ))
}

fn empty_pbr_bake_debug() -> PbrBakeDebug {
    PbrBakeDebug {
        texture_width: 0,
        texture_height: 0,
        uvs: Vec::new(),
        raster_mask: Vec::new(),
        sample_positions: Vec::new(),
        sample_attrs: Vec::new(),
        base_color_float: Vec::new(),
        metallic_float: Vec::new(),
        roughness_float: Vec::new(),
        alpha_float: Vec::new(),
        base_color_rgba_u8: Vec::new(),
        metallic_roughness_u8: Vec::new(),
    }
}

fn dense_voxel_linear_index(x: u32, y: u32, z: u32, spatial: [u32; 3]) -> usize {
    let sx = spatial[0] as usize;
    let sy = spatial[1] as usize;
    let x = x as usize;
    let y = y as usize;
    let z = z as usize;
    (z * sy + y) * sx + x
}

pub(super) fn build_voxel_attr_lookup(
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    spatial: [u32; 3],
) -> Result<VoxelAttrLookup, String> {
    if voxel_coords.len() != voxel_attrs.len() {
        return Err(format!(
            "decode pbr voxel tensor mismatch: coords={} attrs={}",
            voxel_coords.len(),
            voxel_attrs.len()
        ));
    }

    let spatial_x = spatial[0] as usize;
    let spatial_y = spatial[1] as usize;
    let spatial_z = spatial[2] as usize;
    let spatial_cells = spatial_x
        .checked_mul(spatial_y)
        .and_then(|value| value.checked_mul(spatial_z))
        .ok_or_else(|| {
            format!(
                "decode pbr spatial volume overflow: spatial=[{},{},{}]",
                spatial[0], spatial[1], spatial[2]
            )
        })?;

    if spatial_cells <= DENSE_VOXEL_LOOKUP_MAX_CELLS {
        let mut occupancy = vec![0u8; spatial_cells];
        let mut attrs = vec![[0.0f32; 6]; spatial_cells];
        for (coord, value) in voxel_coords.iter().zip(voxel_attrs.iter()) {
            let idx = dense_voxel_linear_index(coord[1], coord[2], coord[3], spatial);
            occupancy[idx] = 255;
            attrs[idx] = *value;
        }
        Ok(VoxelAttrLookup::Dense {
            spatial,
            occupancy,
            attrs,
        })
    } else {
        // Sparse lookup remains the canonical path for large coordinate volumes.
        let mut map: VoxelAttrMap = HashMap::with_capacity_and_hasher(
            voxel_coords.len().saturating_mul(2),
            VoxelAttrFastHasher::default(),
        );
        for (coord, value) in voxel_coords.iter().zip(voxel_attrs.iter()) {
            map.insert(pack_coord(coord[1], coord[2], coord[3]), *value);
        }
        Ok(VoxelAttrLookup::Sparse { spatial, map })
    }
}

fn sample_voxel_attr_from_lookup(
    position: [f32; 3],
    lookup: &VoxelAttrLookup,
) -> Result<Option<[f32; 6]>, String> {
    match lookup {
        VoxelAttrLookup::Dense {
            spatial,
            occupancy,
            attrs,
        } => sample_voxel_attr_dense(position, occupancy.as_slice(), attrs.as_slice(), *spatial),
        VoxelAttrLookup::Sparse { spatial, map } => sample_voxel_attr(position, map, *spatial),
    }
}

#[allow(clippy::type_complexity)]
pub(super) fn bake_pbr_from_voxels_with_options(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    projection_source: Option<PbrProjectionSource<'_>>,
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
    pbr_texture_size: Option<usize>,
    capture_debug: bool,
    prefer_wgpu_sampling: bool,
) -> Result<(PbrBakeMesh, Option<MeshPbrTextures>, Option<PbrBakeDebug>), String> {
    bake_pbr_from_voxels_with_options_and_projection_bvh(
        vertices,
        faces,
        projection_source,
        None,
        voxel_coords,
        voxel_attrs,
        fallback_spatial_resolution,
        pbr_texture_size,
        capture_debug,
        prefer_wgpu_sampling,
    )
}

#[allow(clippy::type_complexity)]
pub(super) fn bake_pbr_from_voxels_with_options_and_projection_bvh(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    projection_source: Option<PbrProjectionSource<'_>>,
    projection_bvh_override: Option<&ProjectionBvh<'_>>,
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
    pbr_texture_size: Option<usize>,
    capture_debug: bool,
    prefer_wgpu_sampling: bool,
) -> Result<(PbrBakeMesh, Option<MeshPbrTextures>, Option<PbrBakeDebug>), String> {
    if vertices.is_empty() || faces.is_empty() {
        return Ok((
            PbrBakeMesh {
                vertices: Vec::new(),
                faces: Vec::new(),
                uvs: Vec::new(),
            },
            None,
            if capture_debug {
                Some(empty_pbr_bake_debug())
            } else {
                None
            },
        ));
    }
    if voxel_coords.len() != voxel_attrs.len() {
        return Err(format!(
            "decode pbr voxel tensor mismatch: coords={} attrs={}",
            voxel_coords.len(),
            voxel_attrs.len()
        ));
    }
    if voxel_coords.is_empty() {
        return Err("decode pbr requires non-empty voxel coordinates".to_string());
    }

    let texture_size = runtime_pbr_texture_size(pbr_texture_size);
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr uv begin (vertices={} faces={} texture_size={})",
        vertices.len(),
        faces.len(),
        texture_size
    ));
    let uv_start = Instant::now();
    let uv_domain = build_uv_raster_domain(vertices, faces, texture_size);
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr uv complete ({:.2} ms, raster_vertices={} raster_faces={})",
        uv_start.elapsed().as_secs_f64() * 1000.0,
        uv_domain.raster_vertices.len(),
        uv_domain.raster_faces.len()
    ));
    let projection_enabled = projection_source.is_some() || projection_bvh_override.is_some();
    let projection_start = Instant::now();
    let mut projection_bvh_owned: Option<ProjectionBvh<'_>> = None;
    if projection_bvh_override.is_none() {
        let source_to_build = projection_source.filter(|source| {
            !source.vertices.is_empty()
                && !source.faces.is_empty()
                && (source.vertices.as_ptr() != vertices.as_ptr()
                    || source.vertices.len() != vertices.len()
                    || source.faces.as_ptr() != faces.as_ptr()
                    || source.faces.len() != faces.len())
        });
        if let Some(source) = source_to_build {
            pbr_stage_log(format!(
                "burn_trellis: decode.pbr projection_bvh begin (vertices={} faces={})",
                source.vertices.len(),
                source.faces.len()
            ));
            projection_bvh_owned = Some(ProjectionBvh::build(source)?);
        }
    }
    let projection_bvh = projection_bvh_override.or(projection_bvh_owned.as_ref());
    if projection_enabled && projection_bvh_override.is_some() {
        pbr_stage_log("burn_trellis: decode.pbr projection_bvh reused (enabled=true)");
    } else if projection_enabled {
        pbr_stage_log(format!(
            "burn_trellis: decode.pbr projection_bvh complete ({:.2} ms, enabled={})",
            projection_start.elapsed().as_secs_f64() * 1000.0,
            projection_bvh.is_some()
        ));
    }
    let texel_count = texture_size * texture_size;
    let mut raster_mask = vec![0u8; texel_count];
    let mut base_color_float = vec![[0.0f32; 4]; texel_count];
    let mut metallic_float = vec![0.0f32; texel_count];
    let mut roughness_float = vec![1.0f32; texel_count];
    let mut alpha_float = vec![0.0f32; texel_count];
    let mut sample_positions = if capture_debug {
        Vec::with_capacity(texel_count / 2)
    } else {
        Vec::new()
    };
    let mut sample_attrs = if capture_debug {
        Vec::with_capacity(texel_count / 2)
    } else {
        Vec::new()
    };

    let mut spatial = [
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
    ];
    for coord in voxel_coords {
        spatial[0] = spatial[0].max(coord[1].saturating_add(1));
        spatial[1] = spatial[1].max(coord[2].saturating_add(1));
        spatial[2] = spatial[2].max(coord[3].saturating_add(1));
    }
    let voxel_lookup = build_voxel_attr_lookup(voxel_coords, voxel_attrs, spatial)?;
    #[cfg(feature = "runtime-model-wgpu")]
    // This path is intentionally opt-in and large-workload-gated: the staged decode
    // pipeline still rasterizes triangles on host, so we only offload dense trilinear
    // sampling when there is enough work to amortize tensor upload/dispatch overhead.
    let use_wgpu_dense_sampling = prefer_wgpu_sampling
        && !capture_debug
        && texel_count >= DENSE_VOXEL_WGPU_SAMPLE_MIN_POSITIONS
        && matches!(voxel_lookup, VoxelAttrLookup::Dense { .. });
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let _ = prefer_wgpu_sampling;
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_positions = Vec::<[f32; 3]>::new();
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_next = Vec::<i32>::new();
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_head = vec![-1i32; texel_count];
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_tail = vec![-1i32; texel_count];
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_candidate_counts = vec![0u8; texel_count];
    #[cfg(feature = "runtime-model-wgpu")]
    let mut deferred_raster_mask = vec![0u8; texel_count];
    #[cfg(feature = "runtime-model-wgpu")]
    if use_wgpu_dense_sampling {
        deferred_positions.reserve(texel_count / 2);
        deferred_next.reserve(texel_count / 2);
    }

    pbr_stage_log(format!(
        "burn_trellis: decode.pbr raster_sample begin (texels={} wgpu_dense={} projection={})",
        texel_count,
        {
            #[cfg(feature = "runtime-model-wgpu")]
            {
                use_wgpu_dense_sampling
            }
            #[cfg(not(feature = "runtime-model-wgpu"))]
            {
                false
            }
        },
        projection_bvh.is_some()
    ));
    let raster_start = Instant::now();
    let mut sample_error: Option<String> = None;
    for face in uv_domain.raster_faces.iter().copied() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= uv_domain.raster_vertices.len()
            || i1 >= uv_domain.raster_vertices.len()
            || i2 >= uv_domain.raster_vertices.len()
        {
            continue;
        }
        if i0 >= uv_domain.raster_uvs.len()
            || i1 >= uv_domain.raster_uvs.len()
            || i2 >= uv_domain.raster_uvs.len()
        {
            continue;
        }
        let p0 = uv_domain.raster_vertices[i0];
        let p1 = uv_domain.raster_vertices[i1];
        let p2 = uv_domain.raster_vertices[i2];
        let uv0 = uv_domain.raster_uvs[i0];
        let uv1 = uv_domain.raster_uvs[i1];
        let uv2 = uv_domain.raster_uvs[i2];
        rasterize_triangle(texture_size, [uv0, uv1, uv2], |x, y, bary| {
            if sample_error.is_some() {
                return;
            }
            let idx = y * texture_size + x;
            if raster_mask[idx] != 0 {
                return;
            }

            let raster_position = [
                p0[0] * bary[0] + p1[0] * bary[1] + p2[0] * bary[2],
                p0[1] * bary[0] + p1[1] * bary[1] + p2[1] * bary[2],
                p0[2] * bary[0] + p1[2] * bary[1] + p2[2] * bary[2],
            ];
            let position =
                projection_bvh.map_or(raster_position, |bvh| bvh.closest_point(raster_position));
            #[cfg(feature = "runtime-model-wgpu")]
            if use_wgpu_dense_sampling {
                deferred_raster_mask[idx] = 255;
                if deferred_candidate_counts[idx] as usize
                    >= DENSE_VOXEL_WGPU_MAX_CANDIDATES_PER_TEXEL
                {
                    return;
                }
                let entry = deferred_positions.len();
                deferred_positions.push(position);
                deferred_next.push(-1);
                if deferred_head[idx] < 0 {
                    deferred_head[idx] = entry as i32;
                } else {
                    let tail = deferred_tail[idx] as usize;
                    deferred_next[tail] = entry as i32;
                }
                deferred_tail[idx] = entry as i32;
                deferred_candidate_counts[idx] = deferred_candidate_counts[idx].saturating_add(1);
                return;
            }
            let attrs = match sample_voxel_attr_from_lookup(position, &voxel_lookup) {
                Ok(Some(attrs)) => attrs,
                Ok(None) => {
                    // Match upstream nvdiffrast/grid_sample semantics: raster coverage
                    // owns the mask; sparse holes sample as zero attrs instead of
                    // expanding the inpaint domain.
                    [0.0; 6]
                }
                Err(err) => {
                    sample_error = Some(err);
                    return;
                }
            };
            if raster_mask[idx] == 0 {
                base_color_float[idx] = [attrs[0], attrs[1], attrs[2], attrs[5]];
                metallic_float[idx] = attrs[3];
                roughness_float[idx] = attrs[4];
                alpha_float[idx] = attrs[5];
                raster_mask[idx] = 255;
            }
            if capture_debug {
                sample_positions.push(position);
                sample_attrs.push(attrs);
            }
        });
    }
    if let Some(err) = sample_error {
        return Err(err);
    }
    #[cfg(feature = "runtime-model-wgpu")]
    if use_wgpu_dense_sampling {
        let wgpu_sampler = match &voxel_lookup {
            VoxelAttrLookup::Dense {
                spatial,
                occupancy,
                attrs,
            } => DenseVoxelWgpuSampler::new(occupancy.as_slice(), attrs.as_slice(), *spatial)?,
            VoxelAttrLookup::Sparse { .. } => {
                return Err(
                    "decode pbr internal error: deferred wgpu sampler requires dense lookup"
                        .to_string(),
                );
            }
        };
        // Resolve texels by trying candidates in first-hit order per texel until one
        // samples successfully; this preserves canonical behavior while avoiding
        // sampling duplicate candidates after a texel is already resolved.
        let mut cursor = deferred_head;
        let mut active_texels = cursor
            .iter()
            .enumerate()
            .filter_map(|(idx, head)| (*head >= 0).then_some(idx))
            .collect::<Vec<_>>();
        while !active_texels.is_empty() {
            let mut next_active_texels = Vec::new();
            let mut start = 0usize;
            while start < active_texels.len() {
                let end = (start + DENSE_VOXEL_WGPU_SAMPLE_BATCH).min(active_texels.len());
                let batch_texels = &active_texels[start..end];
                let mut batch_positions = Vec::with_capacity(batch_texels.len());
                let mut batch_meta = Vec::with_capacity(batch_texels.len());
                for &texel_idx in batch_texels {
                    let entry = cursor[texel_idx];
                    if entry < 0 {
                        continue;
                    }
                    let entry = entry as usize;
                    batch_positions.push(deferred_positions[entry]);
                    batch_meta.push((texel_idx, entry));
                }
                let sampled =
                    sample_voxel_attr_dense_wgpu_batch(batch_positions.as_slice(), &wgpu_sampler)?;
                for (sampled_attrs, (texel_idx, entry)) in
                    sampled.into_iter().zip(batch_meta.into_iter())
                {
                    if raster_mask[texel_idx] != 0 {
                        continue;
                    }
                    if let Some(attrs) = sampled_attrs {
                        base_color_float[texel_idx] = [attrs[0], attrs[1], attrs[2], attrs[5]];
                        metallic_float[texel_idx] = attrs[3];
                        roughness_float[texel_idx] = attrs[4];
                        alpha_float[texel_idx] = attrs[5];
                        raster_mask[texel_idx] = 255;
                    } else {
                        let next = deferred_next[entry];
                        cursor[texel_idx] = next;
                        if next >= 0 {
                            next_active_texels.push(texel_idx);
                        }
                    }
                }
                start = end;
            }
            active_texels = next_active_texels;
        }
        for (idx, covered) in deferred_raster_mask.iter().copied().enumerate() {
            if covered != 0 && raster_mask[idx] == 0 {
                raster_mask[idx] = 255;
            }
        }
    }
    let covered_texels = raster_mask.iter().filter(|value| **value != 0).count();
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr raster_sample complete ({:.2} ms, covered_texels={} coverage={:.4})",
        raster_start.elapsed().as_secs_f64() * 1000.0,
        covered_texels,
        covered_texels as f64 / texel_count.max(1) as f64
    ));

    let inpaint_start = Instant::now();
    inpaint_texture_channels(
        texture_size,
        raster_mask.as_mut_slice(),
        base_color_float.as_mut_slice(),
        metallic_float.as_mut_slice(),
        roughness_float.as_mut_slice(),
        alpha_float.as_mut_slice(),
    )?;
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr inpaint complete ({:.2} ms)",
        inpaint_start.elapsed().as_secs_f64() * 1000.0
    ));

    let pack_start = Instant::now();
    let mut base_color_rgba_u8 = vec![0u8; texel_count * 4];
    let mut metallic_roughness_u8 = vec![0u8; texel_count * 4];
    for idx in 0..texel_count {
        let off = idx * 4;
        let rgba = base_color_float[idx];
        base_color_rgba_u8[off] = quantize_unorm8(rgba[0]);
        base_color_rgba_u8[off + 1] = quantize_unorm8(rgba[1]);
        base_color_rgba_u8[off + 2] = quantize_unorm8(rgba[2]);
        base_color_rgba_u8[off + 3] = quantize_unorm8(alpha_float[idx]);
        metallic_roughness_u8[off] = 0;
        metallic_roughness_u8[off + 1] = quantize_unorm8(roughness_float[idx]);
        metallic_roughness_u8[off + 2] = quantize_unorm8(metallic_float[idx]);
        metallic_roughness_u8[off + 3] = 255;
    }
    pbr_stage_log(format!(
        "burn_trellis: decode.pbr texture_pack complete ({:.2} ms)",
        pack_start.elapsed().as_secs_f64() * 1000.0
    ));

    let debug_base_color_rgba = if capture_debug {
        base_color_rgba_u8.clone()
    } else {
        Vec::new()
    };
    let debug_metallic_roughness = if capture_debug {
        metallic_roughness_u8.clone()
    } else {
        Vec::new()
    };
    let debug_uvs = if capture_debug {
        uv_domain.output_uvs.clone()
    } else {
        Vec::new()
    };
    let pbr_textures = MeshPbrTextures {
        base_color: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: base_color_rgba_u8,
        },
        metallic_roughness: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: metallic_roughness_u8,
        },
        normal: None,
        emissive: None,
        occlusion: None,
    };

    let debug = if capture_debug {
        Some(PbrBakeDebug {
            texture_width: texture_size,
            texture_height: texture_size,
            uvs: debug_uvs,
            raster_mask,
            sample_positions,
            sample_attrs,
            base_color_float,
            metallic_float,
            roughness_float,
            alpha_float,
            base_color_rgba_u8: debug_base_color_rgba,
            metallic_roughness_u8: debug_metallic_roughness,
        })
    } else {
        None
    };

    Ok((
        PbrBakeMesh {
            vertices: uv_domain.output_vertices,
            faces: uv_domain.output_faces,
            uvs: uv_domain.output_uvs,
        },
        Some(pbr_textures),
        debug,
    ))
}

pub(super) fn build_uv_raster_domain(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    texture_size: usize,
) -> UvRasterDomain {
    if vertices.is_empty() || faces.is_empty() || texture_size == 0 {
        return UvRasterDomain {
            output_vertices: Vec::new(),
            output_faces: Vec::new(),
            output_uvs: Vec::new(),
            raster_vertices: Vec::new(),
            raster_uvs: Vec::new(),
            raster_faces: Vec::new(),
        };
    }

    let mut atlas_faces = Vec::with_capacity(faces.len());
    let mut parents = Vec::with_capacity(faces.len());
    let mut edge_owner = fast_hash_map_with_capacity::<(u32, u32), usize>(faces.len() * 3);
    for face in faces.iter().copied() {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let normal = triangle_normal(vertices, face);
        if normal[0] == 0.0 && normal[1] == 0.0 && normal[2] == 0.0 {
            continue;
        }
        let chart = box_atlas_chart(normal);
        let atlas_face_idx = atlas_faces.len();
        atlas_faces.push(UvAtlasFace {
            source: face,
            chart,
            component: usize::MAX,
        });
        parents.push(atlas_face_idx);

        for edge in face_edges(face) {
            if let Some(prev_idx) = edge_owner.insert(edge, atlas_face_idx) {
                if atlas_faces[prev_idx].chart == chart {
                    uv_atlas_union(parents.as_mut_slice(), prev_idx, atlas_face_idx);
                }
            }
        }
    }

    let mut components = Vec::<UvAtlasComponent>::new();
    let mut root_to_component = fast_hash_map_with_capacity::<usize, usize>(atlas_faces.len());
    for face_idx in 0..atlas_faces.len() {
        let root = uv_atlas_find(parents.as_mut_slice(), face_idx);
        let component_idx = *root_to_component.entry(root).or_insert_with(|| {
            let idx = components.len();
            components.push(UvAtlasComponent::new(atlas_faces[face_idx].chart));
            idx
        });
        atlas_faces[face_idx].component = component_idx;
        let face = atlas_faces[face_idx].source;
        components[component_idx].faces.push(face_idx);
        for vertex_idx in face {
            let projected =
                box_atlas_project(atlas_faces[face_idx].chart, vertices[vertex_idx as usize]);
            components[component_idx].include(projected);
        }
    }
    let fragmented_component_limit = texture_size
        .saturating_mul(texture_size)
        .checked_div(16)
        .unwrap_or(0)
        .max(256);
    let dense_face_texel_limit = texture_size
        .saturating_mul(texture_size)
        .saturating_mul(4)
        .max(1);
    if components.len() > fragmented_component_limit || atlas_faces.len() > dense_face_texel_limit {
        pbr_stage_log(format!(
            "burn_trellis: decode.pbr uv using chart atlas fallback (components={} faces={} texture_size={})",
            components.len(),
            atlas_faces.len(),
            texture_size
        ));
        let mut chart_components =
            assign_chart_atlas_components(vertices, atlas_faces.as_mut_slice());
        pack_uv_atlas_components(chart_components.as_mut_slice(), texture_size);
        return assemble_uv_raster_domain(
            vertices,
            atlas_faces.as_slice(),
            chart_components.as_slice(),
            texture_size,
        );
    }

    pack_uv_atlas_components(components.as_mut_slice(), texture_size);
    assemble_uv_raster_domain(
        vertices,
        atlas_faces.as_slice(),
        components.as_slice(),
        texture_size,
    )
}

fn assign_chart_atlas_components(
    vertices: &[[f32; 3]],
    atlas_faces: &mut [UvAtlasFace],
) -> Vec<UvAtlasComponent> {
    let mut components = Vec::<UvAtlasComponent>::new();
    let mut chart_to_component = [usize::MAX; 6];
    for face_idx in 0..atlas_faces.len() {
        let chart = atlas_faces[face_idx]
            .chart
            .min(chart_to_component.len() - 1);
        let component_idx = if chart_to_component[chart] == usize::MAX {
            let idx = components.len();
            chart_to_component[chart] = idx;
            components.push(UvAtlasComponent::new(chart));
            idx
        } else {
            chart_to_component[chart]
        };
        atlas_faces[face_idx].component = component_idx;
        components[component_idx].faces.push(face_idx);
        for vertex_idx in atlas_faces[face_idx].source {
            if let Some(vertex) = vertices.get(vertex_idx as usize) {
                let projected = box_atlas_project(atlas_faces[face_idx].chart, *vertex);
                components[component_idx].include(projected);
            }
        }
    }
    components
}

fn assemble_uv_raster_domain(
    vertices: &[[f32; 3]],
    atlas_faces: &[UvAtlasFace],
    components: &[UvAtlasComponent],
    texture_size: usize,
) -> UvRasterDomain {
    let mut raster_vertices = Vec::with_capacity(atlas_faces.len() * 2);
    let mut raster_uvs = Vec::with_capacity(atlas_faces.len() * 2);
    let mut raster_faces = Vec::with_capacity(atlas_faces.len());
    let mut vertex_map = fast_hash_map_with_capacity::<(usize, u32), u32>(atlas_faces.len() * 2);
    for atlas_face in atlas_faces {
        let mut out_face = [0u32; 3];
        let component = &components[atlas_face.component];
        for (corner, source_idx) in atlas_face.source.iter().copied().enumerate() {
            let out_idx =
                if let Some(existing) = vertex_map.get(&(atlas_face.component, source_idx)) {
                    *existing
                } else {
                    let vertex = vertices[source_idx as usize];
                    let projected = box_atlas_project(atlas_face.chart, vertex);
                    let uv = component.uv(projected, texture_size);
                    let idx = raster_vertices.len() as u32;
                    raster_vertices.push(vertex);
                    raster_uvs.push(uv);
                    vertex_map.insert((atlas_face.component, source_idx), idx);
                    idx
                };
            out_face[corner] = out_idx;
        }
        raster_faces.push(out_face);
    }

    let output_uvs = glb_output_uvs_from_raster_uvs(raster_uvs.as_slice());
    UvRasterDomain {
        output_vertices: raster_vertices.clone(),
        output_faces: raster_faces.clone(),
        output_uvs,
        raster_vertices,
        raster_uvs,
        raster_faces,
    }
}

pub(super) fn glb_output_uvs_from_raster_uvs(raster_uvs: &[[f32; 2]]) -> Vec<[f32; 2]> {
    raster_uvs
        .iter()
        .map(|uv| [uv[0].clamp(0.0, 1.0), 1.0 - uv[1].clamp(0.0, 1.0)])
        .collect()
}

#[derive(Debug, Clone)]
struct UvAtlasFace {
    source: [u32; 3],
    chart: usize,
    component: usize,
}

#[derive(Debug, Clone)]
struct UvAtlasComponent {
    chart: usize,
    min: [f32; 3],
    max: [f32; 3],
    faces: Vec<usize>,
    pos_px: [usize; 2],
    size_px: [usize; 2],
}

impl UvAtlasComponent {
    fn new(chart: usize) -> Self {
        Self {
            chart,
            min: [f32::INFINITY; 3],
            max: [f32::NEG_INFINITY; 3],
            faces: Vec::new(),
            pos_px: [0, 0],
            size_px: [textureless_chart_size(), textureless_chart_size()],
        }
    }

    fn include(&mut self, projected: [f32; 2]) {
        self.min[0] = self.min[0].min(projected[0]);
        self.min[1] = self.min[1].min(projected[1]);
        self.max[0] = self.max[0].max(projected[0]);
        self.max[1] = self.max[1].max(projected[1]);
    }

    fn range(&self) -> [f32; 2] {
        [
            (self.max[0] - self.min[0]).abs().max(1.0e-6),
            (self.max[1] - self.min[1]).abs().max(1.0e-6),
        ]
    }

    fn uv(&self, projected: [f32; 2], texture_size: usize) -> [f32; 2] {
        let range = self.range();
        let local = [
            ((projected[0] - self.min[0]) / range[0]).clamp(0.0, 1.0),
            ((projected[1] - self.min[1]) / range[1]).clamp(0.0, 1.0),
        ];
        let denom = texture_size.saturating_sub(1).max(1) as f32;
        let inner_w = self.size_px[0].saturating_sub(2).max(1) as f32;
        let inner_h = self.size_px[1].saturating_sub(2).max(1) as f32;
        let u_px = self.pos_px[0] as f32 + 1.0 + local[0] * inner_w;
        let v_px = self.pos_px[1] as f32 + 1.0 + local[1] * inner_h;
        [
            (u_px / denom).clamp(0.0, 1.0),
            (v_px / denom).clamp(0.0, 1.0),
        ]
    }
}

fn textureless_chart_size() -> usize {
    3
}

fn pack_uv_atlas_components(components: &mut [UvAtlasComponent], texture_size: usize) {
    if components.is_empty() {
        return;
    }
    let total_area = components
        .iter()
        .map(|component| {
            let range = component.range();
            range[0] * range[1]
        })
        .sum::<f32>()
        .max(1.0e-8);
    let target_area = (texture_size * texture_size) as f32 * PBR_UV_ATLAS_TARGET_OCCUPANCY;
    let mut scale = (target_area / total_area).sqrt();
    for _ in 0..96 {
        if let Some(rects) = try_pack_uv_atlas_components(components, texture_size, scale) {
            for (component, rect) in components.iter_mut().zip(rects.into_iter()) {
                component.pos_px = rect.pos_px;
                component.size_px = rect.size_px;
            }
            return;
        }
        scale *= 0.92;
    }

    // Degenerate fallback: keep every component in-bounds even if the atlas is
    // too fragmented for the requested texture size. This should be rare for
    // TRELLIS.2 meshes after decimation, but it keeps export deterministic.
    let mut cursor = [0usize, 0usize];
    let mut row_h = 0usize;
    for component in components {
        let size = textureless_chart_size();
        if cursor[0] + size > texture_size {
            cursor[0] = 0;
            cursor[1] = cursor[1].saturating_add(row_h);
            row_h = 0;
        }
        component.pos_px = [
            cursor[0].min(texture_size.saturating_sub(1)),
            cursor[1].min(texture_size.saturating_sub(1)),
        ];
        component.size_px = [size.min(texture_size.max(1)), size.min(texture_size.max(1))];
        cursor[0] = cursor[0].saturating_add(size);
        row_h = row_h.max(size);
    }
}

#[derive(Debug, Clone, Copy)]
struct UvAtlasRect {
    pos_px: [usize; 2],
    size_px: [usize; 2],
}

fn try_pack_uv_atlas_components(
    components: &[UvAtlasComponent],
    texture_size: usize,
    scale: f32,
) -> Option<Vec<UvAtlasRect>> {
    let rects = components
        .iter()
        .map(|component| {
            let range = component.range();
            let inner_w = (range[0] * scale).ceil().max(1.0) as usize;
            let inner_h = (range[1] * scale).ceil().max(1.0) as usize;
            [
                inner_w.saturating_add(2).min(texture_size.max(1)),
                inner_h.saturating_add(2).min(texture_size.max(1)),
            ]
        })
        .collect::<Vec<_>>();
    if rects
        .iter()
        .any(|rect| rect[0] > texture_size || rect[1] > texture_size)
    {
        return None;
    }
    let mut order = (0..components.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        rects[*b][1]
            .cmp(&rects[*a][1])
            .then_with(|| rects[*b][0].cmp(&rects[*a][0]))
            .then_with(|| components[*a].chart.cmp(&components[*b].chart))
    });

    let mut packed = vec![
        UvAtlasRect {
            pos_px: [0, 0],
            size_px: [0, 0],
        };
        components.len()
    ];
    let mut x = 0usize;
    let mut y = 0usize;
    let mut row_h = 0usize;
    for idx in order {
        let size = rects[idx];
        if x + size[0] > texture_size {
            x = 0;
            y = y.saturating_add(row_h);
            row_h = 0;
        }
        if y + size[1] > texture_size {
            return None;
        }
        packed[idx] = UvAtlasRect {
            pos_px: [x, y],
            size_px: size,
        };
        x = x.saturating_add(size[0]);
        row_h = row_h.max(size[1]);
    }
    Some(packed)
}

fn face_edges(face: [u32; 3]) -> [(u32, u32); 3] {
    [
        sorted_edge(face[0], face[1]),
        sorted_edge(face[1], face[2]),
        sorted_edge(face[2], face[0]),
    ]
}

fn sorted_edge(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

fn uv_atlas_find(parents: &mut [usize], idx: usize) -> usize {
    let parent = parents[idx];
    if parent == idx {
        idx
    } else {
        let root = uv_atlas_find(parents, parent);
        parents[idx] = root;
        root
    }
}

fn uv_atlas_union(parents: &mut [usize], a: usize, b: usize) {
    let root_a = uv_atlas_find(parents, a);
    let root_b = uv_atlas_find(parents, b);
    if root_a != root_b {
        parents[root_b] = root_a;
    }
}

fn box_atlas_project(chart: usize, vertex: [f32; 3]) -> [f32; 2] {
    match chart {
        0 => [vertex[2], vertex[1]],
        1 => [-vertex[2], vertex[1]],
        2 => [vertex[0], vertex[2]],
        3 => [-vertex[0], vertex[2]],
        4 => [vertex[0], vertex[1]],
        _ => [-vertex[0], vertex[1]],
    }
}

fn box_atlas_chart(normal: [f32; 3]) -> usize {
    let ax = normal[0].abs();
    let ay = normal[1].abs();
    let az = normal[2].abs();
    if ax >= ay && ax >= az {
        if normal[0] >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if normal[1] >= 0.0 { 2 } else { 3 }
    } else if normal[2] >= 0.0 {
        4
    } else {
        5
    }
}

pub(super) fn sample_voxel_attr(
    position: [f32; 3],
    voxel_map: &HashMap<u64, [f32; 6], impl BuildHasher>,
    spatial: [u32; 3],
) -> Result<Option<[f32; 6]>, String> {
    if voxel_map.is_empty() {
        return Err("decode pbr sample requires non-empty voxel map".to_string());
    }
    let map_axis = |value: f32, dim: u32| -> f32 {
        let dim = dim.max(1) as f32;
        // Match TRELLIS2 grid-sample coordinate convention: (pos + 0.5) * resolution.
        // Clamp to the valid sparse lattice extent for robust sampling near boundaries.
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
    let x0 = base[0].clamp(0, max_x) as u32;
    let y0 = base[1].clamp(0, max_y) as u32;
    let z0 = base[2].clamp(0, max_z) as u32;
    let x1 = (base[0] + 1).clamp(0, max_x) as u32;
    let y1 = (base[1] + 1).clamp(0, max_y) as u32;
    let z1 = (base[2] + 1).clamp(0, max_z) as u32;

    let wx0 = 1.0 - frac[0];
    let wy0 = 1.0 - frac[1];
    let wz0 = 1.0 - frac[2];
    let wx1 = frac[0];
    let wy1 = frac[1];
    let wz1 = frac[2];

    let mut accum = [0.0f32; 6];
    let mut weight_sum = 0.0f32;
    let mut accumulate_corner = |x: u32, y: u32, z: u32, weight: f32| {
        if weight <= 0.0 {
            return;
        }
        let key = pack_coord(x, y, z);
        if let Some(attrs) = voxel_map.get(&key) {
            for ch in 0..6 {
                accum[ch] += attrs[ch] * weight;
            }
            weight_sum += weight;
        }
    };
    accumulate_corner(x0, y0, z0, wx0 * wy0 * wz0);
    accumulate_corner(x1, y0, z0, wx1 * wy0 * wz0);
    accumulate_corner(x0, y1, z0, wx0 * wy1 * wz0);
    accumulate_corner(x1, y1, z0, wx1 * wy1 * wz0);
    accumulate_corner(x0, y0, z1, wx0 * wy0 * wz1);
    accumulate_corner(x1, y0, z1, wx1 * wy0 * wz1);
    accumulate_corner(x0, y1, z1, wx0 * wy1 * wz1);
    accumulate_corner(x1, y1, z1, wx1 * wy1 * wz1);

    if weight_sum > 1.0e-8 {
        let inv = 1.0 / weight_sum;
        for value in &mut accum {
            *value *= inv;
        }
        return Ok(Some(accum));
    }
    Ok(None)
}

pub(super) fn sample_voxel_attr_dense(
    position: [f32; 3],
    occupancy: &[u8],
    attrs: &[[f32; 6]],
    spatial: [u32; 3],
) -> Result<Option<[f32; 6]>, String> {
    if occupancy.is_empty() || attrs.is_empty() {
        return Err("decode pbr sample requires non-empty voxel map".to_string());
    }
    if occupancy.len() != attrs.len() {
        return Err(format!(
            "decode pbr dense lookup mismatch: occupancy={} attrs={}",
            occupancy.len(),
            attrs.len()
        ));
    }

    let map_axis = |value: f32, dim: u32| -> f32 {
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
    let x0 = base[0].clamp(0, max_x) as u32;
    let y0 = base[1].clamp(0, max_y) as u32;
    let z0 = base[2].clamp(0, max_z) as u32;
    let x1 = (base[0] + 1).clamp(0, max_x) as u32;
    let y1 = (base[1] + 1).clamp(0, max_y) as u32;
    let z1 = (base[2] + 1).clamp(0, max_z) as u32;

    let wx0 = 1.0 - frac[0];
    let wy0 = 1.0 - frac[1];
    let wz0 = 1.0 - frac[2];
    let wx1 = frac[0];
    let wy1 = frac[1];
    let wz1 = frac[2];

    let mut accum = [0.0f32; 6];
    let mut weight_sum = 0.0f32;
    let mut accumulate_corner = |x: u32, y: u32, z: u32, weight: f32| {
        if weight <= 0.0 {
            return;
        }
        let idx = dense_voxel_linear_index(x, y, z, spatial);
        if occupancy[idx] == 0 {
            return;
        }
        let cell = attrs[idx];
        for ch in 0..6 {
            accum[ch] += cell[ch] * weight;
        }
        weight_sum += weight;
    };
    accumulate_corner(x0, y0, z0, wx0 * wy0 * wz0);
    accumulate_corner(x1, y0, z0, wx1 * wy0 * wz0);
    accumulate_corner(x0, y1, z0, wx0 * wy1 * wz0);
    accumulate_corner(x1, y1, z0, wx1 * wy1 * wz0);
    accumulate_corner(x0, y0, z1, wx0 * wy0 * wz1);
    accumulate_corner(x1, y0, z1, wx1 * wy0 * wz1);
    accumulate_corner(x0, y1, z1, wx0 * wy1 * wz1);
    accumulate_corner(x1, y1, z1, wx1 * wy1 * wz1);

    if weight_sum > 1.0e-8 {
        let inv = 1.0 / weight_sum;
        for value in &mut accum {
            *value *= inv;
        }
        return Ok(Some(accum));
    }
    Ok(None)
}

#[cfg(feature = "runtime-model-wgpu")]
fn sample_voxel_attr_dense_wgpu_batch(
    positions: &[[f32; 3]],
    sampler: &DenseVoxelWgpuSampler,
) -> Result<Vec<Option<[f32; 6]>>, String> {
    if positions.is_empty() {
        return Ok(Vec::new());
    }

    let rows = positions.len();
    let mut positions_flat = Vec::with_capacity(rows.saturating_mul(3));
    for position in positions {
        positions_flat.extend_from_slice(position);
    }
    let positions_t = Tensor::<DefaultWgpuBackend, 2>::from_data(
        TensorData::new(positions_flat, [rows, 3]),
        &sampler.device,
    );
    let sampled_t = dense_trilinear_sample_attrs_wgpu(
        positions_t,
        sampler.occupancy_t.clone(),
        sampler.attrs_t.clone(),
        sampler.spatial,
    )
    .map_err(|err| format!("decode pbr dense wgpu kernel sample failed: {err}"))?;
    let [sample_rows, sample_cols] = sampled_t.dims();
    if sample_rows != rows || sample_cols != 7 {
        return Err(format!(
            "decode pbr dense wgpu sample output dims mismatch: got=[{},{}] expected=[{},7]",
            sample_rows, sample_cols, rows
        ));
    }
    let sampled_flat = sampled_t
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("decode pbr dense wgpu sample extraction failed: {err:?}"))?;
    if sampled_flat.len() != rows.saturating_mul(7) {
        return Err(format!(
            "decode pbr dense wgpu sample output len mismatch: got={} expected={}",
            sampled_flat.len(),
            rows.saturating_mul(7)
        ));
    }

    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row.saturating_mul(7);
        let support = sampled_flat[base + 6];
        if support <= 1.0e-8 {
            out.push(None);
            continue;
        }
        let mut attrs_out = [0.0f32; 6];
        for channel in 0..6 {
            attrs_out[channel] = sampled_flat[base + channel];
        }
        out.push(Some(attrs_out));
    }
    Ok(out)
}

pub(super) fn rasterize_triangle(
    texture_size: usize,
    tri_uv: [[f32; 2]; 3],
    mut draw: impl FnMut(usize, usize, [f32; 3]),
) {
    let to_px = |uv: [f32; 2]| -> [f32; 2] {
        [
            uv[0].clamp(0.0, 1.0) * (texture_size.saturating_sub(1)) as f32,
            (1.0 - uv[1].clamp(0.0, 1.0)) * (texture_size.saturating_sub(1)) as f32,
        ]
    };
    let p0 = to_px(tri_uv[0]);
    let p1 = to_px(tri_uv[1]);
    let p2 = to_px(tri_uv[2]);
    let min_x = p0[0]
        .min(p1[0])
        .min(p2[0])
        .floor()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let max_x = p0[0]
        .max(p1[0])
        .max(p2[0])
        .ceil()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let min_y = p0[1]
        .min(p1[1])
        .min(p2[1])
        .floor()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let max_y = p0[1]
        .max(p1[1])
        .max(p2[1])
        .ceil()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let samples = [
                [0.5f32, 0.5f32],
                [0.2f32, 0.2f32],
                [0.5f32, 0.2f32],
                [0.8f32, 0.2f32],
                [0.2f32, 0.5f32],
                [0.8f32, 0.5f32],
                [0.2f32, 0.8f32],
                [0.5f32, 0.8f32],
                [0.8f32, 0.8f32],
            ];
            for sample in samples {
                let bary = barycentric_2d([x as f32 + sample[0], y as f32 + sample[1]], p0, p1, p2);
                if bary[0] >= -1.0e-6 && bary[1] >= -1.0e-6 && bary[2] >= -1.0e-6 {
                    draw(x, y, bary);
                    break;
                }
            }
        }
    }
}

pub(super) fn barycentric_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [f32; 3] {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1.0e-12 {
        return [-1.0, -1.0, -1.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    [u, v, w]
}

pub(super) fn inpaint_texture_channels(
    texture_size: usize,
    mask: &mut [u8],
    base_color_float: &mut [[f32; 4]],
    metallic_float: &mut [f32],
    roughness_float: &mut [f32],
    alpha_float: &mut [f32],
) -> Result<(), String> {
    let texels = texture_size * texture_size;
    if mask.len() != texels {
        return Err(format!(
            "decode pbr inpaint mask length mismatch: mask={} texels={}",
            mask.len(),
            texels
        ));
    }
    if base_color_float.len() != texels
        || metallic_float.len() != texels
        || roughness_float.len() != texels
        || alpha_float.len() != texels
    {
        return Err(format!(
            "decode pbr inpaint tensor length mismatch: base={} metallic={} roughness={} alpha={} texels={}",
            base_color_float.len(),
            metallic_float.len(),
            roughness_float.len(),
            alpha_float.len(),
            texels
        ));
    }
    if texels == 0 {
        return Err("decode pbr requires non-zero texture size".to_string());
    }
    if mask.iter().all(|value| *value == 0) {
        return Err("decode pbr requires at least one covered texel".to_string());
    }
    // Approximate the upstream Telea inpaint stage with a deterministic nearest
    // covered-texel fill. Keep the original raster mask intact for debug
    // observability, matching OpenCV's input-mask semantics.
    let mut filled = vec![false; texels];
    let mut queue = Vec::with_capacity(texels);
    for idx in 0..texels {
        if mask[idx] != 0 {
            filled[idx] = true;
            queue.push(idx);
        }
    }

    let mut head = 0usize;
    while head < queue.len() {
        let src = queue[head];
        head += 1;
        let x = src % texture_size;
        let y = src / texture_size;

        if x > 0 {
            let dst = src - 1;
            if !filled[dst] {
                copy_inpaint_texel(
                    dst,
                    src,
                    base_color_float,
                    metallic_float,
                    roughness_float,
                    alpha_float,
                );
                filled[dst] = true;
                queue.push(dst);
            }
        }
        if x + 1 < texture_size {
            let dst = src + 1;
            if !filled[dst] {
                copy_inpaint_texel(
                    dst,
                    src,
                    base_color_float,
                    metallic_float,
                    roughness_float,
                    alpha_float,
                );
                filled[dst] = true;
                queue.push(dst);
            }
        }
        if y > 0 {
            let dst = src - texture_size;
            if !filled[dst] {
                copy_inpaint_texel(
                    dst,
                    src,
                    base_color_float,
                    metallic_float,
                    roughness_float,
                    alpha_float,
                );
                filled[dst] = true;
                queue.push(dst);
            }
        }
        if y + 1 < texture_size {
            let dst = src + texture_size;
            if !filled[dst] {
                copy_inpaint_texel(
                    dst,
                    src,
                    base_color_float,
                    metallic_float,
                    roughness_float,
                    alpha_float,
                );
                filled[dst] = true;
                queue.push(dst);
            }
        }
    }
    Ok(())
}

fn copy_inpaint_texel(
    dst: usize,
    src: usize,
    base_color_float: &mut [[f32; 4]],
    metallic_float: &mut [f32],
    roughness_float: &mut [f32],
    alpha_float: &mut [f32],
) {
    base_color_float[dst] = base_color_float[src];
    metallic_float[dst] = metallic_float[src];
    roughness_float[dst] = roughness_float[src];
    alpha_float[dst] = alpha_float[src];
}

pub(super) fn quantize_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(feature = "runtime-model")]
#[allow(dead_code)]
pub(super) fn occupancy_to_coords(
    occupancy: &[f32],
    resolution: usize,
    threshold: f32,
    max_coords: Option<usize>,
) -> Vec<[u32; 4]> {
    let mut candidates = Vec::new();
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (z * resolution + y) * resolution + x;
                let value = occupancy[idx];
                if value > threshold {
                    candidates.push((idx, value));
                }
            }
        }
    }

    if let Some(limit) = max_coords
        && limit > 0
        && candidates.len() > limit
    {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(limit);
        candidates.sort_by_key(|a| a.0);
    }

    let mut coords = Vec::with_capacity(candidates.len());
    for (idx, _) in candidates {
        let x = idx % resolution;
        let y = (idx / resolution) % resolution;
        let z = idx / (resolution * resolution);
        coords.push([0, x as u32, y as u32, z as u32]);
    }
    coords
}

#[cfg(feature = "runtime-model")]
pub(super) fn map_coord_to_dense_flat(
    coord: [u32; 4],
    sparse_resolution: usize,
    dense_resolution: usize,
) -> usize {
    let map_axis = |value: u32| -> usize {
        if sparse_resolution <= 1 || dense_resolution <= 1 {
            return 0;
        }
        let mapped = (value as usize)
            .saturating_mul(dense_resolution)
            .saturating_div(sparse_resolution.max(1));
        mapped.min(dense_resolution - 1)
    };
    let x = map_axis(coord[1]);
    let y = map_axis(coord[2]);
    let z = map_axis(coord[3]);
    (z * dense_resolution + y) * dense_resolution + x
}

pub(super) fn sparse_resolution_for_pipeline(pipeline_type: &str) -> usize {
    match pipeline_type {
        "512" | "512_base" => 32,
        "1024" | "1024_single" => 64,
        "1024_cascade" => 32,
        "1536_cascade" => 32,
        _ => 32,
    }
}

pub(super) fn final_resolution_for_pipeline(pipeline_type: &str) -> usize {
    match pipeline_type {
        "512" | "512_base" => 512,
        "1024" | "1024_single" | "1024_cascade" => 1024,
        "1536_cascade" => 1536,
        _ => 512,
    }
}

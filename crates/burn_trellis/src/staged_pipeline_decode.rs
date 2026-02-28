use super::*;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault};

#[cfg(feature = "runtime-model-wgpu")]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "runtime-model-wgpu")]
use burn_flex_gmm::wgpu::{DefaultWgpuBackend, dense_trilinear_sample_attrs_wgpu};
#[cfg(feature = "runtime-model-wgpu")]
use burn_wgpu::WgpuDevice;
use rustc_hash::FxHasher;
#[cfg(all(
    feature = "uv-xatlas",
    not(target_arch = "wasm32"),
    target_os = "windows"
))]
use xatlas_rs::{ChartOptions, IndexFormat, MeshDecl, PackOptions, Xatlas};

// Internal helpers for TRELLIS staged decode, mesh extraction, and PBR baking.
// Kept in a separate module so `staged_pipeline.rs` stays focused on stage orchestration.

#[derive(Debug, Clone)]
struct UvRasterDomain {
    output_uvs: Vec<[f32; 2]>,
    raster_vertices: Vec<[f32; 3]>,
    raster_uvs: Vec<[f32; 2]>,
    raster_faces: Vec<[u32; 3]>,
}

type VoxelAttrFastHasher = BuildHasherDefault<FxHasher>;
type VoxelAttrMap = HashMap<u64, [f32; 6], VoxelAttrFastHasher>;
// Keep dense lookup bounded to avoid excessive host memory use when sparse
// coords span large volumes; large cases stay on sparse hash lookup.
const DENSE_VOXEL_LOOKUP_MAX_CELLS: usize = 2_500_000;
#[cfg(feature = "runtime-model-wgpu")]
const DENSE_VOXEL_WGPU_SAMPLE_MIN_POSITIONS: usize = 2_048;
#[cfg(feature = "runtime-model-wgpu")]
const DENSE_VOXEL_WGPU_SAMPLE_BATCH: usize = 65_536;

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
    Ok(DecodeShapeSubSample {
        coords,
        feats,
        spatial_shape: sub.spatial_shape,
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

pub(super) fn runtime_pbr_texture_size() -> usize {
    256
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
    let (uvs, textures, debug) = bake_pbr_from_voxels_with_options(
        vertices,
        faces,
        voxel_coords,
        voxel_attrs,
        fallback_spatial_resolution,
        true,
        false,
    )?;
    Ok((uvs, textures, debug.unwrap_or_else(empty_pbr_bake_debug)))
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
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
    capture_debug: bool,
    prefer_wgpu_sampling: bool,
) -> Result<(Vec<[f32; 2]>, Option<MeshPbrTextures>, Option<PbrBakeDebug>), String> {
    if vertices.is_empty() || faces.is_empty() {
        return Ok((
            Vec::new(),
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

    let texture_size = runtime_pbr_texture_size();
    let uv_domain = build_uv_raster_domain(vertices, faces, texture_size);
    let texel_count = texture_size * texture_size;
    let mut raster_mask = vec![0u8; texel_count];
    let mut base_color_float = vec![[0.0f32; 4]; texel_count];
    let mut metallic_float = vec![0.0f32; texel_count];
    let mut roughness_float = vec![1.0f32; texel_count];
    let mut alpha_float = vec![1.0f32; texel_count];
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
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let use_wgpu_dense_sampling = false;
    let mut deferred_texel_indices = Vec::<usize>::new();
    let mut deferred_positions = Vec::<[f32; 3]>::new();
    if use_wgpu_dense_sampling {
        deferred_texel_indices.reserve(texel_count / 2);
        deferred_positions.reserve(texel_count / 2);
    }

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
            if !capture_debug && raster_mask[idx] != 0 {
                return;
            }

            let position = [
                p0[0] * bary[0] + p1[0] * bary[1] + p2[0] * bary[2],
                p0[1] * bary[0] + p1[1] * bary[1] + p2[1] * bary[2],
                p0[2] * bary[0] + p1[2] * bary[1] + p2[2] * bary[2],
            ];
            if use_wgpu_dense_sampling {
                deferred_texel_indices.push(idx);
                deferred_positions.push(position);
                return;
            }
            let attrs = match sample_voxel_attr_from_lookup(position, &voxel_lookup) {
                Ok(Some(attrs)) => attrs,
                Ok(None) => {
                    // Canonical strict behavior: sparse holes are allowed; leave texel uncovered.
                    return;
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
        let mut start = 0usize;
        while start < deferred_positions.len() {
            let end = (start + DENSE_VOXEL_WGPU_SAMPLE_BATCH).min(deferred_positions.len());
            let batch_positions = &deferred_positions[start..end];
            let sampled = sample_voxel_attr_dense_wgpu_batch(batch_positions, &wgpu_sampler)?;
            for (local_idx, sampled_attrs) in sampled.into_iter().enumerate() {
                let stream_idx = start + local_idx;
                let texel_idx = deferred_texel_indices[stream_idx];
                if raster_mask[texel_idx] != 0 {
                    continue;
                }
                let Some(attrs) = sampled_attrs else {
                    continue;
                };
                base_color_float[texel_idx] = [attrs[0], attrs[1], attrs[2], attrs[5]];
                metallic_float[texel_idx] = attrs[3];
                roughness_float[texel_idx] = attrs[4];
                alpha_float[texel_idx] = attrs[5];
                raster_mask[texel_idx] = 255;
            }
            start = end;
        }
    }

    inpaint_texture_channels(
        texture_size,
        raster_mask.as_mut_slice(),
        base_color_float.as_mut_slice(),
        metallic_float.as_mut_slice(),
        roughness_float.as_mut_slice(),
        alpha_float.as_mut_slice(),
    )?;

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

    Ok((uv_domain.output_uvs, Some(pbr_textures), debug))
}

fn build_uv_raster_domain(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    texture_size: usize,
) -> UvRasterDomain {
    #[cfg(not(all(
        feature = "uv-xatlas",
        not(target_arch = "wasm32"),
        target_os = "windows"
    )))]
    let _ = texture_size;

    #[cfg(all(
        feature = "uv-xatlas",
        not(target_arch = "wasm32"),
        target_os = "windows"
    ))]
    if let Some(domain) = build_uv_raster_domain_xatlas(vertices, faces, texture_size) {
        return domain;
    }

    // Portable default path when xatlas is unavailable/disabled.
    let output_uvs = box_uv_unwrap(vertices, faces);
    UvRasterDomain {
        output_uvs: output_uvs.clone(),
        raster_vertices: vertices.to_vec(),
        raster_uvs: output_uvs,
        raster_faces: faces.to_vec(),
    }
}

#[cfg(all(
    feature = "uv-xatlas",
    not(target_arch = "wasm32"),
    target_os = "windows"
))]
fn build_uv_raster_domain_xatlas(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    texture_size: usize,
) -> Option<UvRasterDomain> {
    if vertices.is_empty() || faces.is_empty() || texture_size == 0 {
        return None;
    }

    let mut indices = Vec::with_capacity(faces.len() * 3);
    for face in faces {
        if face.iter().all(|index| (*index as usize) < vertices.len()) {
            indices.extend_from_slice(face);
        }
    }
    if indices.len() < 3 {
        return None;
    }

    let atlas = Xatlas::new();
    let decl = MeshDecl {
        vertex_count: vertices.len() as u32,
        vertex_position_data: as_byte_slice(vertices),
        vertex_position_stride: std::mem::size_of::<[f32; 3]>() as u32,
        index_count: indices.len() as u32,
        index_data: as_byte_slice(indices.as_slice()),
        index_format: IndexFormat::Uint32,
        ..Default::default()
    };
    atlas.add_mesh(&decl);
    let mut pack_options = PackOptions {
        resolution: texture_size as u32,
        max_chart_size: texture_size as u32,
        block_align: false,
        conservative: false,
        padding: 1,
        ..Default::default()
    };
    pack_options.resolution = texture_size as u32;
    atlas.generate_simple(ChartOptions::default(), pack_options);

    let mut atlas = atlas;
    let meshes = atlas.meshes();
    let mesh = meshes.first()?;
    if mesh.vertices.is_empty() || mesh.indices.len() < 3 {
        return None;
    }

    let scale = (texture_size.max(1) as f32).max(1.0);
    let mut raster_vertices = Vec::with_capacity(mesh.vertices.len());
    let mut raster_uvs = Vec::with_capacity(mesh.vertices.len());
    let mut uv_sums = vec![[0.0f32; 2]; vertices.len()];
    let mut uv_counts = vec![0u32; vertices.len()];
    for vertex in mesh.vertices {
        let source_idx = vertex.xref as usize;
        if source_idx >= vertices.len() {
            raster_vertices.push([0.0, 0.0, 0.0]);
            raster_uvs.push([0.0, 0.0]);
            continue;
        }
        let uv = [
            (vertex.uv[0] / scale).clamp(0.0, 1.0),
            (vertex.uv[1] / scale).clamp(0.0, 1.0),
        ];
        raster_vertices.push(vertices[source_idx]);
        raster_uvs.push(uv);
        uv_sums[source_idx][0] += uv[0];
        uv_sums[source_idx][1] += uv[1];
        uv_counts[source_idx] = uv_counts[source_idx].saturating_add(1);
    }

    let mut output_uvs = box_uv_unwrap(vertices, faces);
    for idx in 0..output_uvs.len() {
        if let Some(&count) = uv_counts.get(idx)
            && count > 0
        {
            output_uvs[idx][0] = uv_sums[idx][0] / count as f32;
            output_uvs[idx][1] = uv_sums[idx][1] / count as f32;
        }
    }

    let mut raster_faces = Vec::with_capacity(mesh.indices.len() / 3);
    for tri in mesh.indices.chunks_exact(3) {
        let a = tri[0];
        let b = tri[1];
        let c = tri[2];
        if (a as usize) < raster_vertices.len()
            && (b as usize) < raster_vertices.len()
            && (c as usize) < raster_vertices.len()
        {
            raster_faces.push([a, b, c]);
        }
    }
    if raster_faces.is_empty() {
        return None;
    }

    Some(UvRasterDomain {
        output_uvs,
        raster_vertices,
        raster_uvs,
        raster_faces,
    })
}

#[cfg(all(
    feature = "uv-xatlas",
    not(target_arch = "wasm32"),
    target_os = "windows"
))]
fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    let len = std::mem::size_of_val(slice);
    let ptr = slice.as_ptr() as *const u8;
    // SAFETY: `ptr` originates from `slice`; resulting byte slice has identical lifetime and bounds.
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub(super) fn box_uv_unwrap(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 2]> {
    if vertices.is_empty() {
        return Vec::new();
    }
    let mut normals = vec![[0.0f32; 3]; vertices.len()];
    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let p0 = vertices[i0];
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        let e0 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e1 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ];
        for &idx in &[i0, i1, i2] {
            normals[idx][0] += n[0];
            normals[idx][1] += n[1];
            normals[idx][2] += n[2];
        }
    }
    let mut min = vertices[0];
    let mut max = vertices[0];
    for vertex in vertices.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let range_x = (max[0] - min[0]).abs().max(1.0e-6);
    let range_y = (max[1] - min[1]).abs().max(1.0e-6);
    let range_z = (max[2] - min[2]).abs().max(1.0e-6);
    vertices
        .iter()
        .zip(normals.iter())
        .map(|(vertex, normal)| {
            // Dominant-axis box projection based on local normal direction.
            let ax = normal[0].abs();
            let ay = normal[1].abs();
            let az = normal[2].abs();
            if ay >= ax && ay >= az {
                let mut u = ((vertex[0] - min[0]) / range_x).clamp(0.0, 1.0);
                if normal[1] < 0.0 {
                    u = 1.0 - u;
                }
                [u, ((vertex[2] - min[2]) / range_z).clamp(0.0, 1.0)]
            } else if ax >= az {
                let mut u = ((vertex[2] - min[2]) / range_z).clamp(0.0, 1.0);
                if normal[0] < 0.0 {
                    u = 1.0 - u;
                }
                [u, ((vertex[1] - min[1]) / range_y).clamp(0.0, 1.0)]
            } else {
                let mut u = ((vertex[0] - min[0]) / range_x).clamp(0.0, 1.0);
                if normal[2] < 0.0 {
                    u = 1.0 - u;
                }
                [u, ((vertex[1] - min[1]) / range_y).clamp(0.0, 1.0)]
            }
        })
        .collect()
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
            let bary = barycentric_2d([x as f32 + 0.5, y as f32 + 0.5], p0, p1, p2);
            if bary[0] >= -1.0e-6 && bary[1] >= -1.0e-6 && bary[2] >= -1.0e-6 {
                draw(x, y, bary);
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
    // Canonical strict behavior: do not perform nearest-neighbor rescue/inpaint for uncovered texels.
    // Leave untouched texels as-is (alpha stays 0 from initialization), and keep mask for observability.
    Ok(())
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

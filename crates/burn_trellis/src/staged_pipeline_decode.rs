use super::*;
use std::collections::VecDeque;

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(feature = "runtime-model")]
pub(super) fn runtime_subdivision_to_sample(sub: &SparseSubdivisionLogits) -> DecodeShapeSubSample {
    let mut feats = Vec::with_capacity(sub.coords.len());
    for row_idx in 0..sub.coords.len() {
        let mut row = [0.0f32; 8];
        let base = row_idx * 8;
        if base + 8 <= sub.logits.len() {
            row.copy_from_slice(&sub.logits[base..base + 8]);
        }
        feats.push(row);
    }
    DecodeShapeSubSample {
        coords: sub.coords.clone(),
        feats,
        spatial_shape: sub.spatial_shape,
    }
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
    std::env::var("TRELLIS2_PBR_TEX_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 64)
        .unwrap_or(256)
}

#[allow(clippy::type_complexity)]
pub(super) fn bake_pbr_from_voxels(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
) -> (Vec<[f32; 2]>, Option<MeshPbrTextures>, PbrBakeDebug) {
    if vertices.is_empty() || faces.is_empty() {
        return (
            Vec::new(),
            None,
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
            },
        );
    }

    let texture_size = runtime_pbr_texture_size();
    let uv_domain = build_uv_raster_domain(vertices, faces, texture_size);
    let texel_count = texture_size * texture_size;
    let mut raster_mask = vec![0u8; texel_count];
    let mut base_color_float = vec![[0.0f32; 4]; texel_count];
    let mut metallic_float = vec![0.0f32; texel_count];
    let mut roughness_float = vec![1.0f32; texel_count];
    let mut alpha_float = vec![1.0f32; texel_count];
    let mut sample_positions = Vec::with_capacity(texel_count / 2);
    let mut sample_attrs = Vec::with_capacity(texel_count / 2);

    let mut voxel_map = HashMap::with_capacity(voxel_coords.len().saturating_mul(2));
    let mut spatial = [
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
    ];
    for (idx, coord) in voxel_coords.iter().enumerate() {
        let attrs = voxel_attrs
            .get(idx)
            .copied()
            .unwrap_or([0.5, 0.5, 0.5, 0.0, 1.0, 1.0]);
        voxel_map.insert(pack_coord(coord[1], coord[2], coord[3]), attrs);
        spatial[0] = spatial[0].max(coord[1].saturating_add(1));
        spatial[1] = spatial[1].max(coord[2].saturating_add(1));
        spatial[2] = spatial[2].max(coord[3].saturating_add(1));
    }
    let fallback_attr = summarize_voxel_attr(voxel_attrs);

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
            let position = [
                p0[0] * bary[0] + p1[0] * bary[1] + p2[0] * bary[2],
                p0[1] * bary[0] + p1[1] * bary[1] + p2[1] * bary[2],
                p0[2] * bary[0] + p1[2] * bary[1] + p2[2] * bary[2],
            ];
            let attrs =
                sample_voxel_attr(position, &voxel_map, fallback_attr, spatial, voxel_coords);
            let idx = y * texture_size + x;
            if raster_mask[idx] == 0 {
                base_color_float[idx] = [attrs[0], attrs[1], attrs[2], attrs[5]];
                metallic_float[idx] = attrs[3];
                roughness_float[idx] = attrs[4];
                alpha_float[idx] = attrs[5];
                raster_mask[idx] = 255;
            }
            sample_positions.push(position);
            sample_attrs.push(attrs);
        });
    }

    inpaint_texture_channels(
        texture_size,
        raster_mask.as_mut_slice(),
        base_color_float.as_mut_slice(),
        metallic_float.as_mut_slice(),
        roughness_float.as_mut_slice(),
        alpha_float.as_mut_slice(),
        fallback_attr,
    );

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

    let pbr_textures = MeshPbrTextures {
        base_color: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: base_color_rgba_u8.clone(),
        },
        metallic_roughness: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: metallic_roughness_u8.clone(),
        },
        normal: None,
        emissive: None,
        occlusion: None,
    };

    (
        uv_domain.output_uvs.clone(),
        Some(pbr_textures),
        PbrBakeDebug {
            texture_width: texture_size,
            texture_height: texture_size,
            uvs: uv_domain.output_uvs,
            raster_mask,
            sample_positions,
            sample_attrs,
            base_color_float,
            metallic_float,
            roughness_float,
            alpha_float,
            base_color_rgba_u8,
            metallic_roughness_u8,
        },
    )
}

fn build_uv_raster_domain(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    texture_size: usize,
) -> UvRasterDomain {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(domain) = build_uv_raster_domain_xatlas(vertices, faces, texture_size) {
        return domain;
    }

    let output_uvs = box_uv_unwrap(vertices, faces);
    UvRasterDomain {
        output_uvs: output_uvs.clone(),
        raster_vertices: vertices.to_vec(),
        raster_uvs: output_uvs,
        raster_faces: faces.to_vec(),
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

pub(super) fn summarize_voxel_attr(voxel_attrs: &[[f32; 6]]) -> [f32; 6] {
    if voxel_attrs.is_empty() {
        return [0.7, 0.7, 0.7, 0.0, 0.8, 1.0];
    }
    let mut accum = [0.0f32; 6];
    for attrs in voxel_attrs {
        for idx in 0..6 {
            accum[idx] += attrs[idx];
        }
    }
    let inv = 1.0 / voxel_attrs.len() as f32;
    for value in &mut accum {
        *value *= inv;
    }
    accum
}

pub(super) fn sample_voxel_attr(
    position: [f32; 3],
    voxel_map: &HashMap<u64, [f32; 6]>,
    fallback: [f32; 6],
    spatial: [u32; 3],
    voxel_coords: &[[u32; 4]],
) -> [f32; 6] {
    if voxel_map.is_empty() {
        return fallback;
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

    let mut accum = [0.0f32; 6];
    let mut weight_sum = 0.0f32;
    for dz in 0..=1 {
        for dy in 0..=1 {
            for dx in 0..=1 {
                let x = base[0] + dx;
                let y = base[1] + dy;
                let z = base[2] + dz;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let x = x.min(spatial[0] as i32 - 1) as u32;
                let y = y.min(spatial[1] as i32 - 1) as u32;
                let z = z.min(spatial[2] as i32 - 1) as u32;
                let wx = if dx == 0 { 1.0 - frac[0] } else { frac[0] };
                let wy = if dy == 0 { 1.0 - frac[1] } else { frac[1] };
                let wz = if dz == 0 { 1.0 - frac[2] } else { frac[2] };
                let weight = wx * wy * wz;
                let key = pack_coord(x, y, z);
                if let Some(attrs) = voxel_map.get(&key) {
                    for ch in 0..6 {
                        accum[ch] += attrs[ch] * weight;
                    }
                    weight_sum += weight;
                }
            }
        }
    }
    if weight_sum > 1.0e-8 {
        let inv = 1.0 / weight_sum;
        for value in &mut accum {
            *value *= inv;
        }
        return accum;
    }

    let nearest = [
        coord[0].round() as i32,
        coord[1].round() as i32,
        coord[2].round() as i32,
    ];
    let key = pack_coord(
        nearest[0].clamp(0, spatial[0] as i32 - 1) as u32,
        nearest[1].clamp(0, spatial[1] as i32 - 1) as u32,
        nearest[2].clamp(0, spatial[2] as i32 - 1) as u32,
    );
    if let Some(attrs) = voxel_map.get(&key) {
        return *attrs;
    }

    let mut best = None;
    let mut best_dist = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = nearest[0] + dx;
                let y = nearest[1] + dy;
                let z = nearest[2] + dz;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let key = pack_coord(x as u32, y as u32, z as u32);
                if let Some(attrs) = voxel_map.get(&key) {
                    let dist = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some(*attrs);
                    }
                }
            }
        }
    }
    if let Some(attrs) = best {
        return attrs;
    }
    if !voxel_coords.is_empty() {
        // Last-resort stable fallback for sparse misses: nearest known coordinate.
        let mut nearest_idx = 0usize;
        let mut nearest_dist = f32::INFINITY;
        for (idx, coord) in voxel_coords.iter().enumerate() {
            let dx = coord[1] as f32 - base[0] as f32;
            let dy = coord[2] as f32 - base[1] as f32;
            let dz = coord[3] as f32 - base[2] as f32;
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_idx = idx;
            }
        }
        if let Some(attrs) = voxel_map.get(&pack_coord(
            voxel_coords[nearest_idx][1],
            voxel_coords[nearest_idx][2],
            voxel_coords[nearest_idx][3],
        )) {
            return *attrs;
        }
    }
    fallback
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
    fallback: [f32; 6],
) {
    let texels = texture_size * texture_size;
    if mask.len() != texels {
        return;
    }
    let neighbors = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    let mut nearest = vec![usize::MAX; texels];
    let mut queue = VecDeque::with_capacity(texels);
    for idx in 0..texels {
        if mask[idx] != 0 {
            nearest[idx] = idx;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        let seed = nearest[idx];
        let x = idx % texture_size;
        let y = idx / texture_size;
        for (dx, dy) in neighbors {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if nx < 0 || ny < 0 || nx >= texture_size as isize || ny >= texture_size as isize {
                continue;
            }
            let nidx = ny as usize * texture_size + nx as usize;
            if nearest[nidx] == usize::MAX {
                nearest[nidx] = seed;
                queue.push_back(nidx);
            }
        }
    }

    for idx in 0..texels {
        if mask[idx] != 0 {
            continue;
        }
        let source = nearest[idx];
        if source != usize::MAX {
            base_color_float[idx] = base_color_float[source];
            metallic_float[idx] = metallic_float[source];
            roughness_float[idx] = roughness_float[source];
            alpha_float[idx] = alpha_float[source];
        } else {
            base_color_float[idx] = [fallback[0], fallback[1], fallback[2], fallback[5]];
            metallic_float[idx] = fallback[3];
            roughness_float[idx] = fallback[4];
            alpha_float[idx] = fallback[5];
        }
        mask[idx] = 255;
    }
}

pub(super) fn quantize_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub(super) fn occupancy_target(preprocess: &PreprocessOutput, resolution: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; resolution * resolution * resolution];
    for z in 0..resolution {
        let z_norm = z as f32 / (resolution.saturating_sub(1).max(1) as f32);
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (z * resolution + y) * resolution + x;
                let luma = sample_pixel_luma(preprocess, x as u32, y as u32, z as u32);
                let depth_bias = 1.0 - (z_norm - 0.5).abs() * 1.6;
                out[idx] = (luma * depth_bias).clamp(0.0, 1.0);
            }
        }
    }
    out
}

#[cfg(feature = "runtime-model")]
pub(super) fn build_sparse_cond_from_preprocess(
    preprocess: &PreprocessOutput,
    tokens: usize,
    cond_channels: usize,
) -> Vec<f32> {
    let patch_side = (tokens as f32).sqrt().floor().max(1.0) as usize;
    let patch_tokens = (patch_side * patch_side).min(tokens);
    let extra_tokens = tokens.saturating_sub(patch_tokens);
    let width = preprocess.width.max(1) as usize;
    let height = preprocess.height.max(1) as usize;
    let mut out = Vec::with_capacity(tokens * cond_channels);
    for token_idx in 0..tokens {
        let (x, y, extra_scale) = if token_idx < patch_tokens {
            let x = token_idx % patch_side;
            let y = token_idx / patch_side;
            (x, y, 0.0f32)
        } else {
            let extra_idx = token_idx - patch_tokens;
            let x = width / 2;
            let y = height / 2;
            let scale = if extra_tokens > 0 {
                extra_idx as f32 / extra_tokens as f32
            } else {
                0.0
            };
            (x, y, scale)
        };
        let xx = if token_idx < patch_tokens {
            (x * width / patch_side).min(width - 1)
        } else {
            x.min(width - 1)
        };
        let yy = if token_idx < patch_tokens {
            (y * height / patch_side).min(height - 1)
        } else {
            y.min(height - 1)
        };
        let offset = (yy * width + xx) * 3;
        let r = preprocess.rgb[offset] as f32 / 255.0;
        let g = preprocess.rgb[offset + 1] as f32 / 255.0;
        let b = preprocess.rgb[offset + 2] as f32 / 255.0;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let nx = if patch_side > 1 {
            x as f32 / (patch_side as f32 - 1.0)
        } else {
            0.0
        };
        let ny = if patch_side > 1 {
            y as f32 / (patch_side as f32 - 1.0)
        } else {
            0.0
        };
        let basis = [r, g, b, luma, nx, ny, extra_scale];
        for channel in 0..cond_channels {
            let base = basis[channel % basis.len()];
            let gain = 1.0 + ((channel / basis.len()) % 17) as f32 / 17.0;
            let phase = ((token_idx + channel + 1) as f32 * 0.013).sin();
            out.push((base * gain + 0.1 * phase).clamp(-1.0, 1.0));
        }
    }
    out
}

pub(super) fn latent_to_occupancy(latent: &[f32], channels: usize, resolution: usize) -> Vec<f32> {
    let voxels = resolution * resolution * resolution;
    let mut occupancy = vec![0.0f32; voxels];
    for idx in 0..voxels {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += latent[ch * voxels + idx];
        }
        occupancy[idx] = sum / channels.max(1) as f32;
    }
    // Map to [0, 1] using per-sample dynamic normalization.
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &occupancy {
        min = min.min(*value);
        max = max.max(*value);
    }
    let denom = (max - min).max(1.0e-6);
    for value in &mut occupancy {
        *value = (*value - min) / denom;
    }
    occupancy
}

pub(super) fn upsample_occupancy(input: &[f32], input_res: usize, output_res: usize) -> Vec<f32> {
    if input_res == output_res {
        return input.to_vec();
    }
    let mut out = vec![0.0f32; output_res * output_res * output_res];
    for z in 0..output_res {
        let src_z = z * input_res / output_res;
        for y in 0..output_res {
            let src_y = y * input_res / output_res;
            for x in 0..output_res {
                let src_x = x * input_res / output_res;
                let src_idx = (src_z * input_res + src_y) * input_res + src_x;
                let dst_idx = (z * output_res + y) * output_res + x;
                out[dst_idx] = input[src_idx];
            }
        }
    }
    out
}

#[cfg(feature = "runtime-model")]
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
        candidates.sort_by(|a, b| a.0.cmp(&b.0));
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
pub(super) fn runtime_max_sparse_coords() -> Option<usize> {
    std::env::var("TRELLIS2_MAX_SPARSE_COORDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
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

pub(super) fn sample_pixel_luma(preprocess: &PreprocessOutput, x: u32, y: u32, z: u32) -> f32 {
    let width = preprocess.width.max(1);
    let height = preprocess.height.max(1);
    let xx = (x as usize * width as usize / 32).min(width as usize - 1);
    let yy = (y as usize * height as usize / 32).min(height as usize - 1);
    let offset = (yy * width as usize + xx) * 3;
    let r = preprocess.rgb[offset] as f32 / 255.0;
    let g = preprocess.rgb[offset + 1] as f32 / 255.0;
    let b = preprocess.rgb[offset + 2] as f32 / 255.0;
    let z_mod = 0.9 + 0.2 * ((z as f32 / 31.0) - 0.5);
    (0.2126 * r + 0.7152 * g + 0.0722 * b) * z_mod
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

pub(super) fn canonical_cube() -> Mesh {
    let vertices = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    Mesh {
        vertices,
        faces,
        uvs: Vec::new(),
        material: None,
        pbr_textures: None,
    }
}

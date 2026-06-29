#[derive(Default)]
struct DecodeHookOverrides<'a> {
    decode_shape_subs: Option<&'a [DecodeShapeSubSample]>,
    decode_tex_voxels: Option<&'a DecodeTexVoxelSample>,
    decode_mesh_vertices: Option<&'a [[f32; 3]]>,
    decode_mesh_faces: Option<&'a [[u32; 3]]>,
}

#[cfg(feature = "runtime-model")]
#[derive(Clone, Copy, Default)]
struct RuntimeDecodeModels<'a> {
    shape_decoder: Option<&'a FdgDecoderRuntime>,
    tex_decoder: Option<&'a SparseUnetVaeDecoderRuntime>,
}

#[allow(clippy::too_many_arguments)]
fn decode_latent_to_outputs(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    final_resolution_override: Option<usize>,
    target_faces: Option<usize>,
    pbr_texture_size: Option<usize>,
    parity_strict: bool,
    capture_debug_artifacts: bool,
    decode_overrides: DecodeHookOverrides<'_>,
    decode_output_mode: TrellisDecodeOutputMode,
    #[cfg(feature = "runtime-model")] runtime_decoders: RuntimeDecodeModels<'_>,
) -> Result<DecodedLatentOutput, String> {
    let has_decode_override = decode_overrides.decode_shape_subs.is_some()
        || decode_overrides.decode_tex_voxels.is_some()
        || decode_overrides.decode_mesh_vertices.is_some()
        || decode_overrides.decode_mesh_faces.is_some();
    if has_decode_override {
        return Err(
            "burn_trellis: decode hook override tensors are disabled on canonical runtime decode path"
                .to_string(),
        );
    }

    #[cfg(feature = "runtime-model")]
    {
        let Some(shape_decoder) = runtime_decoders.shape_decoder else {
            return Err(
                "burn_trellis: shape runtime decoder is required (missing `shape_slat_decoder` runtime)"
                    .to_string(),
            );
        };
        let tex_decoder = if decode_output_mode.needs_texture_attrs() {
            Some(runtime_decoders.tex_decoder.ok_or_else(|| {
                "burn_trellis: tex runtime decoder is required for textured TRELLIS decode (missing `tex_slat_decoder` runtime)"
                    .to_string()
            })?)
        } else {
            runtime_decoders.tex_decoder
        };
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (
                shape,
                tex,
                pipeline_type,
                final_resolution_override,
                target_faces,
                pbr_texture_size,
                parity_strict,
                capture_debug_artifacts,
                decode_output_mode,
                shape_decoder,
                tex_decoder,
            );
            return Err(
                "burn_trellis: sync runtime decode is unsupported on wasm; use async runtime decode"
                    .to_string(),
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let decoded = pollster::block_on(decode_latent_with_runtime_decoders(
                shape,
                tex,
                RuntimeDecodeRequest {
                    pipeline_type,
                    final_resolution: final_resolution_override
                        .unwrap_or_else(|| final_resolution_for_pipeline(pipeline_type)),
                    target_faces,
                    pbr_texture_size,
                    parity_strict,
                    capture_debug_artifacts,
                    decode_output_mode,
                    shape_decoder,
                    tex_decoder,
                    shape_guide_subdivisions: None,
                },
            ))
            .map_err(|err| format!("burn_trellis: runtime decode pipeline failed: {err}"))?;
            let _ = decode_overrides;
            Ok(decoded)
        }
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (
            shape,
            tex,
            pipeline_type,
            final_resolution_override,
            target_faces,
            pbr_texture_size,
            parity_strict,
            capture_debug_artifacts,
            decode_overrides,
            decode_output_mode,
        );
        Err("burn_trellis: TRELLIS decode requires `runtime-model` feature".to_string())
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[cfg(feature = "runtime-model")]
async fn decode_latent_to_outputs_async(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    final_resolution_override: Option<usize>,
    target_faces: Option<usize>,
    pbr_texture_size: Option<usize>,
    parity_strict: bool,
    capture_debug_artifacts: bool,
    decode_overrides: DecodeHookOverrides<'_>,
    decode_output_mode: TrellisDecodeOutputMode,
    runtime_decoders: RuntimeDecodeModels<'_>,
) -> Result<DecodedLatentOutput, String> {
    let has_decode_override = decode_overrides.decode_shape_subs.is_some()
        || decode_overrides.decode_tex_voxels.is_some()
        || decode_overrides.decode_mesh_vertices.is_some()
        || decode_overrides.decode_mesh_faces.is_some();
    if has_decode_override {
        return Err(
            "burn_trellis: decode hook override tensors are disabled on canonical runtime decode path"
                .to_string(),
        );
    }

    let Some(shape_decoder) = runtime_decoders.shape_decoder else {
        return Err(
            "burn_trellis: shape runtime decoder is required (missing `shape_slat_decoder` runtime)"
                .to_string(),
        );
    };
    let tex_decoder = if decode_output_mode.needs_texture_attrs() {
        Some(runtime_decoders.tex_decoder.ok_or_else(|| {
            "burn_trellis: tex runtime decoder is required for textured TRELLIS decode (missing `tex_slat_decoder` runtime)"
                .to_string()
        })?)
    } else {
        runtime_decoders.tex_decoder
    };

    let decoded = decode_latent_with_runtime_decoders(
        shape,
        tex,
        RuntimeDecodeRequest {
            pipeline_type,
            final_resolution: final_resolution_override
                .unwrap_or_else(|| final_resolution_for_pipeline(pipeline_type)),
            target_faces,
            pbr_texture_size,
            parity_strict,
            capture_debug_artifacts,
            decode_output_mode,
            shape_decoder,
            tex_decoder,
            shape_guide_subdivisions: None,
        },
    )
    .await
    .map_err(|err| format!("burn_trellis: runtime decode pipeline failed: {err}"))?;
    let _ = decode_overrides;
    Ok(decoded)
}

#[cfg(feature = "runtime-model")]
fn merge_voxel_attrs_for_decode(
    shape_coords: &[[u32; 4]],
    tex_coords: &[[u32; 4]],
    tex_attrs: &[[f32; 6]],
    _parity_strict: bool,
) -> Result<Vec<[f32; 6]>, String> {
    if tex_coords.len() != tex_attrs.len() {
        return Err(format!(
            "decode tex voxel output mismatch: coords={} attrs={}",
            tex_coords.len(),
            tex_attrs.len()
        ));
    }
    if shape_coords.is_empty() {
        return Ok(Vec::new());
    }

    if shape_coords.len() == tex_coords.len() && shape_coords == tex_coords {
        return Ok(tex_attrs.to_vec());
    }
    Err(format!(
        "decode tex voxel coords differ from shape coords (shape_rows={} tex_rows={})",
        shape_coords.len(),
        tex_coords.len()
    ))
}

#[cfg(feature = "runtime-model")]
type MeshVertices = Vec<[f32; 3]>;
#[cfg(feature = "runtime-model")]
type MeshFaces = Vec<[u32; 3]>;
#[cfg(feature = "runtime-model")]
type MeshSanitizeResult = (MeshVertices, MeshFaces, usize, usize, usize);
#[cfg(feature = "runtime-model")]
const NATIVE_PBR_OVOXEL_REMESH_BAND: f32 = 1.0;
#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
const NATIVE_PBR_SMALL_COMPONENT_AREA: f32 = 1.0e-5;
#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
const NATIVE_PBR_SMALL_COMPONENT_AREA_FRACTION: f32 = 1.0e-4;
#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
const NATIVE_PBR_MAX_HOLE_PERIMETER: f32 = 3.0e-2;

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
type MeshCleanupFastHasher = std::hash::BuildHasherDefault<rustc_hash::FxHasher>;
#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
type MeshCleanupHashMap<K, V> = std::collections::HashMap<K, V, MeshCleanupFastHasher>;
#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
type MeshCleanupHashSet<K> = std::collections::HashSet<K, MeshCleanupFastHasher>;

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn mesh_cleanup_hash_map_with_capacity<K, V>(capacity: usize) -> MeshCleanupHashMap<K, V> {
    MeshCleanupHashMap::with_capacity_and_hasher(
        capacity,
        std::hash::BuildHasherDefault::<rustc_hash::FxHasher>::default(),
    )
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn mesh_cleanup_hash_set_with_capacity<K>(capacity: usize) -> MeshCleanupHashSet<K> {
    MeshCleanupHashSet::with_capacity_and_hasher(
        capacity,
        std::hash::BuildHasherDefault::<rustc_hash::FxHasher>::default(),
    )
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
#[derive(Debug, Clone, Copy, Default)]
struct MeshCleanupStats {
    duplicate_faces_removed: usize,
    nonmanifold_faces_removed: usize,
    small_component_faces_removed: usize,
    hole_faces_added: usize,
}

#[cfg(feature = "runtime-model")]
fn sanitize_mesh_geometry(vertices: MeshVertices, faces: MeshFaces) -> MeshSanitizeResult {
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut sanitized_vertices = Vec::with_capacity(vertices.len());
    for (idx, vertex) in vertices.into_iter().enumerate() {
        if vertex.iter().all(|component| component.is_finite()) {
            remap[idx] = sanitized_vertices.len() as u32;
            sanitized_vertices.push(vertex);
        }
    }
    let dropped_vertices = remap.iter().filter(|&&mapped| mapped == u32::MAX).count();

    let mut sanitized_faces = Vec::with_capacity(faces.len());
    let mut dropped_invalid_faces = 0usize;
    let mut dropped_degenerate_faces = 0usize;
    for [a, b, c] in faces {
        let map_index = |value: u32| -> Option<u32> {
            let idx = value as usize;
            if idx >= remap.len() {
                return None;
            }
            let mapped = remap[idx];
            (mapped != u32::MAX).then_some(mapped)
        };
        let Some(a_mapped) = map_index(a) else {
            dropped_invalid_faces += 1;
            continue;
        };
        let Some(b_mapped) = map_index(b) else {
            dropped_invalid_faces += 1;
            continue;
        };
        let Some(c_mapped) = map_index(c) else {
            dropped_invalid_faces += 1;
            continue;
        };
        if a_mapped == b_mapped || b_mapped == c_mapped || a_mapped == c_mapped {
            dropped_degenerate_faces += 1;
            continue;
        }
        sanitized_faces.push([a_mapped, b_mapped, c_mapped]);
    }

    (
        sanitized_vertices,
        sanitized_faces,
        dropped_vertices,
        dropped_invalid_faces,
        dropped_degenerate_faces,
    )
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn cleanup_mesh_topology(vertices: &mut MeshVertices, faces: &mut MeshFaces) -> MeshCleanupStats {
    let cleanup_start = Instant::now();
    let duplicate_start = Instant::now();
    let duplicate_faces_removed = remove_duplicate_mesh_faces(faces);
    let duplicate_ms = duplicate_start.elapsed().as_secs_f64() * 1000.0;
    let nonmanifold_start = Instant::now();
    let nonmanifold_faces_removed = remove_nonmanifold_mesh_faces(faces);
    let nonmanifold_ms = nonmanifold_start.elapsed().as_secs_f64() * 1000.0;
    let component_start = Instant::now();
    let small_component_faces_removed = remove_small_connected_components(
        vertices.as_slice(),
        faces,
        NATIVE_PBR_SMALL_COMPONENT_AREA,
    );
    let component_ms = component_start.elapsed().as_secs_f64() * 1000.0;
    let hole_start = Instant::now();
    let hole_faces_added =
        fill_small_boundary_holes(vertices, faces, NATIVE_PBR_MAX_HOLE_PERIMETER);
    let hole_ms = hole_start.elapsed().as_secs_f64() * 1000.0;
    let compact_start = Instant::now();
    remove_unreferenced_vertices(vertices, faces);
    let compact_ms = compact_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: mesh topology cleanup passes complete ({:.2} ms, duplicate={:.2} ms nonmanifold={:.2} ms components={:.2} ms holes={:.2} ms compact={:.2} ms faces={} vertices={})",
        cleanup_start.elapsed().as_secs_f64() * 1000.0,
        duplicate_ms,
        nonmanifold_ms,
        component_ms,
        hole_ms,
        compact_ms,
        faces.len(),
        vertices.len()
    );
    MeshCleanupStats {
        duplicate_faces_removed,
        nonmanifold_faces_removed,
        small_component_faces_removed,
        hole_faces_added,
    }
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn remove_duplicate_mesh_faces(faces: &mut MeshFaces) -> usize {
    let before = faces.len();
    let mut seen = mesh_cleanup_hash_set_with_capacity::<[u32; 3]>(faces.len());
    faces.retain(|face| {
        let mut key = *face;
        key.sort_unstable();
        if key[0] == key[1] || key[1] == key[2] {
            return false;
        }
        seen.insert(key)
    });
    before.saturating_sub(faces.len())
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
#[derive(Debug, Clone, Copy)]
struct EdgeFaceOwners {
    second: Option<usize>,
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
#[derive(Debug, Clone, Copy)]
struct BoundaryEdgeOwner {
    count: u8,
    a: u32,
    b: u32,
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn remove_nonmanifold_mesh_faces(faces: &mut MeshFaces) -> usize {
    if faces.is_empty() {
        return 0;
    }

    let mut edge_faces =
        mesh_cleanup_hash_map_with_capacity::<(u32, u32), EdgeFaceOwners>(faces.len() * 3);
    let mut drop_face = vec![false; faces.len()];
    for (face_idx, face) in faces.iter().copied().enumerate() {
        for edge in face_edges_u32(face) {
            match edge_faces.get_mut(&edge) {
                Some(owners) if owners.second.is_none() => {
                    owners.second = Some(face_idx);
                }
                Some(_owners) => {
                    drop_face[face_idx] = true;
                }
                None => {
                    edge_faces.insert(
                        edge,
                        EdgeFaceOwners {
                            second: None,
                        },
                    );
                }
            }
        }
    }

    if !drop_face.iter().any(|drop| *drop) {
        return 0;
    }

    let before = faces.len();
    faces.retain_with_index(|face_idx, _face| !drop_face[face_idx]);
    before.saturating_sub(faces.len())
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn remove_small_connected_components(
    vertices: &[[f32; 3]],
    faces: &mut MeshFaces,
    min_area: f32,
) -> usize {
    if faces.is_empty() {
        return 0;
    }

    let mut parent = (0..faces.len()).collect::<Vec<_>>();
    let mut edge_owner = mesh_cleanup_hash_map_with_capacity::<(u32, u32), usize>(faces.len() * 3);
    for (face_idx, face) in faces.iter().copied().enumerate() {
        for edge in face_edges_u32(face) {
            if let Some(prev_idx) = edge_owner.insert(edge, face_idx) {
                mesh_union(parent.as_mut_slice(), prev_idx, face_idx);
            }
        }
    }

    let mut component_area = vec![0.0f32; faces.len()];
    for (face_idx, face) in faces.iter().copied().enumerate() {
        let root = mesh_find(parent.as_mut_slice(), face_idx);
        let area = triangle_area_by_indices(vertices, face).unwrap_or(0.0);
        component_area[root] += area;
    }
    let total_area = component_area.iter().copied().sum::<f32>();
    let min_area = min_area.max(total_area * NATIVE_PBR_SMALL_COMPONENT_AREA_FRACTION);
    let largest_component = component_area
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(root, _area)| root);

    let before = faces.len();
    faces.retain_with_index(|face_idx, _face| {
        let root = mesh_find(parent.as_mut_slice(), face_idx);
        Some(root) == largest_component || component_area[root] >= min_area
    });
    before.saturating_sub(faces.len())
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn fill_small_boundary_holes(
    vertices: &mut MeshVertices,
    faces: &mut MeshFaces,
    max_hole_perimeter: f32,
) -> usize {
    if faces.is_empty() || vertices.is_empty() {
        return 0;
    }

    let mut edge_counts =
        mesh_cleanup_hash_map_with_capacity::<(u32, u32), BoundaryEdgeOwner>(faces.len() * 3);
    for face in faces.iter().copied() {
        for [a, b] in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
            let key = sorted_edge_u32(a, b);
            edge_counts
                .entry(key)
                .and_modify(|owner| {
                    owner.count = owner.count.saturating_add(1);
                })
                .or_insert(BoundaryEdgeOwner { count: 1, a, b });
        }
    }

    let mut outgoing = mesh_cleanup_hash_map_with_capacity::<u32, Vec<u32>>(faces.len() / 8);
    for owner in edge_counts.into_values() {
        if owner.count == 1 {
            outgoing.entry(owner.a).or_default().push(owner.b);
        }
    }
    for next in outgoing.values_mut() {
        next.sort_unstable();
    }

    let mut used = mesh_cleanup_hash_set_with_capacity::<(u32, u32)>(outgoing.len());
    let mut loops = Vec::<Vec<u32>>::new();
    let mut starts = outgoing.keys().copied().collect::<Vec<_>>();
    starts.sort_unstable();
    for start in starts {
        let Some(candidates) = outgoing.get(&start) else {
            continue;
        };
        let next_candidates = candidates.clone();
        for first_next in next_candidates {
            if used.contains(&(start, first_next)) {
                continue;
            }
            let mut loop_vertices = vec![start];
            let mut current = start;
            let mut next = first_next;
            let mut closed = false;
            for _ in 0..vertices.len().saturating_add(1) {
                if !used.insert((current, next)) {
                    break;
                }
                current = next;
                if current == start {
                    closed = true;
                    break;
                }
                loop_vertices.push(current);
                let Some(options) = outgoing.get(&current) else {
                    break;
                };
                let Some(candidate) = options
                    .iter()
                    .copied()
                    .find(|candidate| !used.contains(&(current, *candidate)))
                else {
                    break;
                };
                next = candidate;
            }
            if closed && loop_vertices.len() >= 3 {
                loops.push(loop_vertices);
            }
        }
    }

    let mut added = 0usize;
    for boundary in loops {
        let perimeter = boundary_loop_perimeter(vertices.as_slice(), boundary.as_slice());
        if !perimeter.is_finite() || perimeter > max_hole_perimeter {
            continue;
        }
        let mut center = [0.0f32; 3];
        let mut valid = true;
        for &idx in &boundary {
            let Some(vertex) = vertices.get(idx as usize) else {
                valid = false;
                break;
            };
            center[0] += vertex[0];
            center[1] += vertex[1];
            center[2] += vertex[2];
        }
        if !valid {
            continue;
        }
        let denom = boundary.len().max(1) as f32;
        center[0] /= denom;
        center[1] /= denom;
        center[2] /= denom;
        let center_idx = vertices.len() as u32;
        vertices.push(center);
        for i in 0..boundary.len() {
            let a = boundary[i];
            let b = boundary[(i + 1) % boundary.len()];
            faces.push([a, center_idx, b]);
            added += 1;
        }
    }
    added
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn remove_unreferenced_vertices(vertices: &mut MeshVertices, faces: &mut MeshFaces) {
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut compacted = Vec::with_capacity(vertices.len());
    for face in faces.iter() {
        for &idx in face {
            let idx_usize = idx as usize;
            if idx_usize >= vertices.len() || remap[idx_usize] != u32::MAX {
                continue;
            }
            remap[idx_usize] = compacted.len() as u32;
            compacted.push(vertices[idx_usize]);
        }
    }
    for face in faces.iter_mut() {
        for idx in face {
            *idx = remap[*idx as usize];
        }
    }
    *vertices = compacted;
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn cleanup_pbr_bake_mesh_topology(mesh: &mut PbrBakeMesh) -> MeshCleanupStats {
    let duplicate_faces_removed = remove_duplicate_mesh_faces(&mut mesh.faces);
    let nonmanifold_faces_removed = remove_nonmanifold_mesh_faces(&mut mesh.faces);
    remove_unreferenced_vertices_uvs_normals(
        &mut mesh.vertices,
        &mut mesh.faces,
        &mut mesh.uvs,
        &mut mesh.normals,
    );
    MeshCleanupStats {
        duplicate_faces_removed,
        nonmanifold_faces_removed,
        small_component_faces_removed: 0,
        hole_faces_added: 0,
    }
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn remove_unreferenced_vertices_uvs_normals(
    vertices: &mut MeshVertices,
    faces: &mut MeshFaces,
    uvs: &mut Vec<[f32; 2]>,
    normals: &mut Vec<[f32; 3]>,
) {
    let keep_uvs = uvs.len() == vertices.len();
    let keep_normals = normals.len() == vertices.len();
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut compacted_vertices = Vec::with_capacity(vertices.len());
    let mut compacted_uvs = Vec::with_capacity(if keep_uvs { vertices.len() } else { 0 });
    let mut compacted_normals = Vec::with_capacity(if keep_normals { vertices.len() } else { 0 });
    for face in faces.iter() {
        for &idx in face {
            let idx_usize = idx as usize;
            if idx_usize >= vertices.len() || remap[idx_usize] != u32::MAX {
                continue;
            }
            remap[idx_usize] = compacted_vertices.len() as u32;
            compacted_vertices.push(vertices[idx_usize]);
            if keep_uvs {
                compacted_uvs.push(uvs[idx_usize]);
            }
            if keep_normals {
                compacted_normals.push(normals[idx_usize]);
            }
        }
    }
    for face in faces.iter_mut() {
        for idx in face {
            *idx = remap[*idx as usize];
        }
    }
    *vertices = compacted_vertices;
    if keep_uvs {
        *uvs = compacted_uvs;
    }
    if keep_normals {
        *normals = compacted_normals;
    } else {
        normals.clear();
    }
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn boundary_loop_perimeter(vertices: &[[f32; 3]], boundary: &[u32]) -> f32 {
    if boundary.len() < 2 {
        return 0.0;
    }
    let mut perimeter = 0.0f32;
    for i in 0..boundary.len() {
        let Some(a) = vertices.get(boundary[i] as usize) else {
            return f32::INFINITY;
        };
        let Some(b) = vertices.get(boundary[(i + 1) % boundary.len()] as usize) else {
            return f32::INFINITY;
        };
        perimeter += distance3(*a, *b);
    }
    perimeter
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn triangle_area_by_indices(vertices: &[[f32; 3]], face: [u32; 3]) -> Option<f32> {
    let a = *vertices.get(face[0] as usize)?;
    let b = *vertices.get(face[1] as usize)?;
    let c = *vertices.get(face[2] as usize)?;
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    Some(0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt())
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn distance3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn face_edges_u32(face: [u32; 3]) -> [(u32, u32); 3] {
    [
        sorted_edge_u32(face[0], face[1]),
        sorted_edge_u32(face[1], face[2]),
        sorted_edge_u32(face[2], face[0]),
    ]
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn sorted_edge_u32(a: u32, b: u32) -> (u32, u32) {
    if a <= b { (a, b) } else { (b, a) }
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn mesh_find(parent: &mut [usize], idx: usize) -> usize {
    let root = parent[idx];
    if root == idx {
        idx
    } else {
        let resolved = mesh_find(parent, root);
        parent[idx] = resolved;
        resolved
    }
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
fn mesh_union(parent: &mut [usize], a: usize, b: usize) {
    let root_a = mesh_find(parent, a);
    let root_b = mesh_find(parent, b);
    if root_a != root_b {
        parent[root_b] = root_a;
    }
}

#[cfg(feature = "runtime-model")]
trait RetainWithIndex<T> {
    fn retain_with_index<F>(&mut self, f: F)
    where
        F: FnMut(usize, &T) -> bool;
}

#[cfg(feature = "runtime-model")]
impl<T> RetainWithIndex<T> for Vec<T> {
    fn retain_with_index<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, &T) -> bool,
    {
        let mut idx = 0usize;
        self.retain(|item| {
            let keep = f(idx, item);
            idx += 1;
            keep
        });
    }
}

#[cfg(feature = "runtime-model")]
fn apply_native_pbr_ovoxel_remesh_domain_scale(mesh: &mut Mesh, final_resolution: u32) {
    let resolution = final_resolution.max(1) as f32;
    let scale = (resolution + 3.0 * NATIVE_PBR_OVOXEL_REMESH_BAND) / resolution;
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }
    for vertex in &mut mesh.vertices {
        vertex[0] *= scale;
        vertex[1] *= scale;
        vertex[2] *= scale;
    }
}

#[cfg(feature = "runtime-model")]
fn decimate_mesh_for_face_budget(
    vertices: &mut Vec<[f32; 3]>,
    faces: &mut Vec<[u32; 3]>,
    target_faces: usize,
) -> Result<(), String> {
    if target_faces == 0 || faces.len() <= target_faces || faces.is_empty() || vertices.is_empty() {
        return Ok(());
    }

    let mut indices = Vec::with_capacity(faces.len() * 3);
    for face in faces.iter() {
        indices.push(face[0]);
        indices.push(face[1]);
        indices.push(face[2]);
    }
    let target_index_count = (target_faces.saturating_mul(3)).min(indices.len());
    if target_index_count < 3 {
        return Err("target face count too small for runtime decimation".to_string());
    }

    let vertices_bytes = meshopt::typed_to_bytes(vertices.as_slice());
    let adapter =
        meshopt::VertexDataAdapter::new(vertices_bytes, std::mem::size_of::<[f32; 3]>(), 0)
            .map_err(|err| format!("meshopt vertex adapter: {err}"))?;

    let mut simplified = Vec::<u32>::new();
    for error_limit in [0.02f32, 0.05, 0.1, 0.25, 0.5, 1.0] {
        let mut stage_error = 0.0f32;
        let candidate = meshopt::simplify(
            &indices,
            &adapter,
            target_index_count,
            error_limit,
            meshopt::SimplifyOptions::None,
            Some(&mut stage_error),
        );
        if candidate.len() < 3 {
            continue;
        }
        simplified = candidate;
        if simplified.len() <= target_index_count {
            break;
        }
    }
    if simplified.len() < 3 {
        return Err("meshopt simplification produced empty mesh".to_string());
    }
    if simplified.len() > target_index_count {
        trellis_stage_log!(
            "burn_trellis: runtime decode decimation preserved topology over exact face budget (target_faces={} produced_faces={})",
            target_faces,
            simplified.len() / 3
        );
    }

    let (vertex_count, remap) =
        meshopt::generate_vertex_remap(vertices.as_slice(), Some(&simplified));
    let remapped_vertices = meshopt::remap_vertex_buffer(vertices.as_slice(), vertex_count, &remap);
    let remapped_indices = meshopt::remap_index_buffer(Some(&simplified), vertex_count, &remap);
    if remapped_indices.len() < 3 {
        return Err("meshopt remap produced empty mesh".to_string());
    }
    let remapped_faces = remapped_indices.as_chunks::<3>().0.to_vec();

    *vertices = remapped_vertices;
    *faces = remapped_faces;
    Ok(())
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
#[derive(Debug, Clone)]
pub struct NativePbrPostprocessInput {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub voxel_coords: Vec<[u32; 4]>,
    pub voxel_attrs: Vec<[f32; 6]>,
    pub final_resolution: u32,
    pub target_faces: Option<usize>,
    pub pbr_texture_size: Option<usize>,
    pub remesh_threads: Option<usize>,
}

#[cfg(all(feature = "runtime-model", not(target_arch = "wasm32")))]
pub fn native_pbr_mesh_from_decoded_tensors(
    input: NativePbrPostprocessInput,
) -> Result<Mesh, String> {
    let NativePbrPostprocessInput {
        vertices,
        faces,
        voxel_coords,
        voxel_attrs,
        final_resolution,
        target_faces,
        pbr_texture_size,
        remesh_threads,
    } = input;
    if voxel_coords.len() != voxel_attrs.len() {
        return Err(format!(
            "rust o_voxel pbr postprocess voxel mismatch: coords={} attrs={}",
            voxel_coords.len(),
            voxel_attrs.len()
        ));
    }

    let (
        mut vertices,
        mut faces,
        dropped_vertices,
        dropped_invalid_faces,
        dropped_degenerate_faces,
    ) = sanitize_mesh_geometry(vertices, faces);
    if dropped_vertices > 0 || dropped_invalid_faces > 0 || dropped_degenerate_faces > 0 {
        trellis_stage_log!(
            "burn_trellis: rust_ovoxel pbr postprocess mesh sanitized (dropped_vertices={} dropped_invalid_faces={} dropped_degenerate_faces={})",
            dropped_vertices,
            dropped_invalid_faces,
            dropped_degenerate_faces
        );
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err("rust o_voxel pbr postprocess received empty mesh after sanitize".to_string());
    }
    let cleanup_stats = cleanup_mesh_topology(&mut vertices, &mut faces);
    if cleanup_stats.duplicate_faces_removed > 0
        || cleanup_stats.nonmanifold_faces_removed > 0
        || cleanup_stats.small_component_faces_removed > 0
        || cleanup_stats.hole_faces_added > 0
    {
        trellis_stage_log!(
            "burn_trellis: rust_ovoxel pbr postprocess mesh cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} small_component_faces_removed={} hole_faces_added={} vertices={} faces={})",
            cleanup_stats.duplicate_faces_removed,
            cleanup_stats.nonmanifold_faces_removed,
            cleanup_stats.small_component_faces_removed,
            cleanup_stats.hole_faces_added,
            vertices.len(),
            faces.len()
        );
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err("rust o_voxel pbr postprocess received empty mesh after cleanup".to_string());
    }

    let projection_vertices = vertices.clone();
    let projection_faces = faces.clone();
    let remesh_start = Instant::now();
    let before_remesh_faces = faces.len();
    let projection_bvh_start = Instant::now();
    let projection_bvh_for_pbr = build_projection_bvh_for_pbr(PbrProjectionSource {
        vertices: projection_vertices.as_slice(),
        faces: projection_faces.as_slice(),
    })?;
    let remesh_bvh_ms = projection_bvh_start.elapsed().as_secs_f64() * 1000.0;
    let remesh_output = remesh_narrow_band_simple_dc_with_projection_bvh(
        &projection_bvh_for_pbr,
        final_resolution.max(1),
        NATIVE_PBR_OVOXEL_REMESH_BAND,
        remesh_threads,
    )?;
    vertices = remesh_output.vertices;
    faces = remesh_output.faces;
    let remesh_cleanup_start = Instant::now();
    let remesh_cleanup_stats = cleanup_mesh_topology(&mut vertices, &mut faces);
    let remesh_cleanup_ms = remesh_cleanup_start.elapsed().as_secs_f64() * 1000.0;
    if remesh_cleanup_stats.duplicate_faces_removed > 0
        || remesh_cleanup_stats.nonmanifold_faces_removed > 0
        || remesh_cleanup_stats.small_component_faces_removed > 0
        || remesh_cleanup_stats.hole_faces_added > 0
    {
        trellis_stage_log!(
            "burn_trellis: rust_ovoxel pbr postprocess remesh cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} small_component_faces_removed={} hole_faces_added={} vertices={} faces={})",
            remesh_cleanup_stats.duplicate_faces_removed,
            remesh_cleanup_stats.nonmanifold_faces_removed,
            remesh_cleanup_stats.small_component_faces_removed,
            remesh_cleanup_stats.hole_faces_added,
            vertices.len(),
            faces.len()
        );
    }
    trellis_stage_log!(
        "burn_trellis: rust_ovoxel pbr postprocess remesh complete ({:.2} ms, bvh={:.2} ms refine={:.2} ms dc={:.2} ms cleanup={:.2} ms active_voxels={} grid_vertices={} from_faces={} to_faces={} vertices={})",
        remesh_start.elapsed().as_secs_f64() * 1000.0,
        remesh_bvh_ms,
        remesh_output.refine_ms,
        remesh_output.dc_ms,
        remesh_cleanup_ms,
        remesh_output.active_voxels,
        remesh_output.grid_vertices,
        before_remesh_faces,
        faces.len(),
        vertices.len()
    );
    let mut use_projection_source = true;
    if let Some(target_faces) = target_faces.filter(|limit| *limit > 0)
        && faces.len() > target_faces
    {
        let before_faces = faces.len();
        decimate_mesh_for_face_budget(&mut vertices, &mut faces, target_faces)?;
        use_projection_source = true;
        let decimate_cleanup_stats = cleanup_mesh_topology(&mut vertices, &mut faces);
        if decimate_cleanup_stats.duplicate_faces_removed > 0
            || decimate_cleanup_stats.nonmanifold_faces_removed > 0
            || decimate_cleanup_stats.small_component_faces_removed > 0
            || decimate_cleanup_stats.hole_faces_added > 0
        {
            trellis_stage_log!(
                "burn_trellis: rust_ovoxel pbr postprocess decimation cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} small_component_faces_removed={} hole_faces_added={} vertices={} faces={})",
                decimate_cleanup_stats.duplicate_faces_removed,
                decimate_cleanup_stats.nonmanifold_faces_removed,
                decimate_cleanup_stats.small_component_faces_removed,
                decimate_cleanup_stats.hole_faces_added,
                vertices.len(),
                faces.len()
            );
        }
        trellis_stage_log!(
            "burn_trellis: rust_ovoxel pbr postprocess decimation complete (target_faces={} from_faces={} to_faces={})",
            target_faces,
            before_faces,
            faces.len()
        );
    }

    let (mut pbr_mesh, pbr_textures, _) = bake_pbr_from_voxels_with_options_and_projection_bvh(
        vertices.as_slice(),
        faces.as_slice(),
        use_projection_source.then_some(PbrProjectionSource {
            vertices: projection_vertices.as_slice(),
            faces: projection_faces.as_slice(),
        }),
        use_projection_source.then_some(&projection_bvh_for_pbr),
        voxel_coords.as_slice(),
        voxel_attrs.as_slice(),
        final_resolution.max(1),
        pbr_texture_size,
        false,
        false,
    )?;
    let pbr_cleanup_stats = cleanup_pbr_bake_mesh_topology(&mut pbr_mesh);
    if pbr_cleanup_stats.duplicate_faces_removed > 0
        || pbr_cleanup_stats.nonmanifold_faces_removed > 0
    {
        trellis_stage_log!(
            "burn_trellis: rust_ovoxel pbr postprocess output cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} vertices={} faces={} uvs={})",
            pbr_cleanup_stats.duplicate_faces_removed,
            pbr_cleanup_stats.nonmanifold_faces_removed,
            pbr_mesh.vertices.len(),
            pbr_mesh.faces.len(),
            pbr_mesh.uvs.len()
        );
    }
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mut mesh = Mesh {
        vertices: pbr_mesh.vertices,
        faces: pbr_mesh.faces,
        uvs: pbr_mesh.uvs,
        normals: pbr_mesh.normals,
        material,
        pbr_textures,
    };
    crate::mesh::remap_mesh_to_python_glb_frame(&mut mesh);
    crate::mesh::orient_mesh_faces_to_positive_volume(&mut mesh);
    Ok(mesh)
}

#[cfg(feature = "runtime-model")]
struct RuntimeDecodeRequest<'a> {
    pipeline_type: &'a str,
    final_resolution: usize,
    target_faces: Option<usize>,
    pbr_texture_size: Option<usize>,
    parity_strict: bool,
    capture_debug_artifacts: bool,
    decode_output_mode: TrellisDecodeOutputMode,
    shape_decoder: &'a FdgDecoderRuntime,
    tex_decoder: Option<&'a SparseUnetVaeDecoderRuntime>,
    shape_guide_subdivisions: Option<&'a [SparseSubdivisionLogits]>,
}

#[cfg(feature = "runtime-model")]
async fn decode_latent_with_runtime_decoders(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    request: RuntimeDecodeRequest<'_>,
) -> Result<DecodedLatentOutput, String> {
    let RuntimeDecodeRequest {
        pipeline_type,
        final_resolution,
        target_faces,
        pbr_texture_size,
        parity_strict,
        capture_debug_artifacts,
        decode_output_mode,
        shape_decoder,
        tex_decoder,
        shape_guide_subdivisions,
    } = request;
    let stage_debug = runtime_stage_debug_enabled();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_coords_wgpu = shape.coords_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_coords_wgpu = tex.coords_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_features_wgpu = shape.features_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    let tex_features_wgpu = tex.features_wgpu.as_ref().cloned();
    #[cfg(feature = "runtime-model-wgpu")]
    // Runtime decode mode must follow tensor residency, not compile-time cfg:
    // strict canonical WGPU stays fail-fast, while explicit host decode runs
    // still work when crate is built with runtime-model-wgpu enabled.
    let using_device_decode_inputs = runtime_decode_uses_device_inputs(
        shape_coords_wgpu.is_some(),
        tex_coords_wgpu.is_some(),
        shape_features_wgpu.is_some(),
        tex_features_wgpu.is_some(),
    );
    #[cfg(feature = "runtime-model-wgpu")]
    let decode_stage_fenced = !using_device_decode_inputs || runtime_stage_fence_enabled();
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let decode_stage_fenced = true;
    let shape_coord_rows = if !shape.coords.is_empty() {
        shape.coords.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            shape_coords_wgpu
                .as_ref()
                .map(|coords_t| coords_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let tex_coord_rows = if !tex.coords.is_empty() {
        tex.coords.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            tex_coords_wgpu
                .as_ref()
                .map(|coords_t| coords_t.dims()[0])
                .unwrap_or(shape_coord_rows)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            shape_coord_rows
        }
    };
    let shape_feature_rows = if !shape.features.is_empty() {
        shape.features.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            shape_features_wgpu
                .as_ref()
                .map(|rows_t| rows_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let tex_feature_rows = if !tex.features.is_empty() {
        tex.features.len()
    } else {
        #[cfg(feature = "runtime-model-wgpu")]
        {
            tex_features_wgpu
                .as_ref()
                .map(|rows_t| rows_t.dims()[0])
                .unwrap_or(0)
        }
        #[cfg(not(feature = "runtime-model-wgpu"))]
        {
            0
        }
    };
    let needs_tex_decode = decode_output_mode.needs_texture_attrs();
    let count = if needs_tex_decode {
        shape_coord_rows
            .min(tex_coord_rows)
            .min(shape_feature_rows)
            .min(tex_feature_rows)
    } else {
        shape_coord_rows.min(shape_feature_rows)
    };
    if count == 0 {
        return Err(if needs_tex_decode {
            "runtime decode received empty shape/tex latent rows".to_string()
        } else {
            "runtime decode received empty shape latent rows".to_string()
        });
    }
    if shape_decoder.out_channels() < 7 {
        return Err(format!(
            "decoder channel mismatch: shape_out={}",
            shape_decoder.out_channels()
        ));
    }
    if let Some(tex_decoder) = tex_decoder
        && tex_decoder.out_channels() < 6
    {
        return Err(format!(
            "decoder channel mismatch: tex_out={}",
            tex_decoder.out_channels()
        ));
    }
    trellis_stage_log!("burn_trellis: stage decode.shape_decoder begin (rows={count})");
    if stage_debug {
        trellis_stage_log!("burn_trellis: decode runtime begin (rows={count})");
    }
    let conv_telemetry_debug = runtime_decoder_conv_telemetry_enabled();
    let shape_rows_host = if shape.features.len() >= count {
        Some(&shape.features[..count])
    } else {
        None
    };
    let tex_rows_host = if tex.features.len() >= count {
        Some(&tex.features[..count])
    } else {
        None
    };
    let shape_coords_host = if !shape.coords.is_empty() {
        Some(&shape.coords[..count])
    } else {
        None
    };
    reset_decoder_conv_telemetry();
    reset_decoder_op_telemetry();
    reset_neighbor_build_stats();
    let shape_decode_start = Instant::now();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_decode_result = if using_device_decode_inputs {
        let shape_coords_wgpu_for_decode = if let Some(coords_t) = shape_coords_wgpu.as_ref() {
            let [rows, cols] = coords_t.dims();
            if cols != 4 {
                return Err(format!(
                    "runtime decode shape coord tensor must have 4 columns, got {}",
                    cols
                ));
            }
            if rows < count {
                return Err(format!(
                    "runtime decode shape coord tensor rows {} smaller than requested count {}",
                    rows, count
                ));
            }
            if rows == count {
                coords_t.clone()
            } else {
                coords_t.clone().slice([0..count, 0..4])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device shape coords; host decode fallback is disabled"
                    .to_string(),
            );
        };
        let shape_rows_wgpu_for_decode = if let Some(rows_t) = shape_features_wgpu.as_ref() {
            let [rows, cols] = rows_t.dims();
            if cols != 32 {
                return Err(format!(
                    "runtime decode shape feature tensor must have 32 columns, got {}",
                    cols
                ));
            }
            if rows == count {
                rows_t.clone()
            } else {
                rows_t.clone().slice([0..count, 0..32])
            }
        } else {
            return Err(
                "runtime decode canonical wgpu path requires device shape rows; host row tensorization fallback is disabled"
                    .to_string(),
            );
        };
        let rows_t = shape_rows_wgpu_for_decode.clone();
        match shape_guide_subdivisions {
            Some(guides) => {
                shape_decoder
                    .decode_with_guidance_result_with_tensors_async(
                        shape_coords_wgpu_for_decode.clone(),
                        rows_t,
                        guides,
                    )
                    .await
            }
            None => {
                shape_decoder
                    .decode_sparse_result_with_tensors_async(
                        shape_coords_wgpu_for_decode.clone(),
                        rows_t,
                    )
                    .await
            }
        }
    } else if let Some(coords_host) = shape_coords_host {
        let shape_rows = shape_rows_host.ok_or_else(|| {
            "runtime decode missing shape host rows for host coord decode path".to_string()
        })?;
        match shape_guide_subdivisions {
            Some(guides) => {
                shape_decoder.decode_with_guidance_result(coords_host, shape_rows, guides)
            }
            None => shape_decoder.decode_sparse_result(coords_host, shape_rows),
        }
    } else {
        return Err("runtime decode missing shape coords for host decode path".to_string());
    };
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let shape_decode_result = if let Some(coords_host) = shape_coords_host {
        let shape_rows = shape_rows_host.ok_or_else(|| {
            "runtime decode missing shape host rows for host coord decode path".to_string()
        })?;
        match shape_guide_subdivisions {
            Some(guides) => {
                shape_decoder.decode_with_guidance_result(coords_host, shape_rows, guides)
            }
            None => shape_decoder.decode_sparse_result(coords_host, shape_rows),
        }
    } else {
        return Err(
            "runtime decode missing shape coords and crate was built without runtime-model-wgpu"
                .to_string(),
        );
    };
    let shape_decode_result = match shape_decode_result {
        Ok(decoded) => decoded,
        Err(err) => {
            trellis_stage_log!("burn_trellis: stage decode.shape_decoder error ({err})");
            return Err(format!("shape runtime decoder failed: {err}"));
        }
    };
    #[cfg(feature = "runtime-model-wgpu")]
    runtime_decode_stage_boundary_sync(
        "shape_decoder",
        using_device_decode_inputs && decode_stage_fenced,
    )?;
    let shape_decoder_ms = shape_decode_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: stage decode.shape_decoder complete ({shape_decoder_ms:.2} ms, subs={}, coords={})",
        shape_decode_result.subdivisions.len(),
        shape_decode_result.rows()
    );
    let shape_conv_telemetry = decoder_conv_telemetry();
    let shape_op_telemetry = decoder_op_telemetry();
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime shape-decoder complete ({:.2} ms, subs={}, coords={})",
            shape_decoder_ms,
            shape_decode_result.subdivisions.len(),
            shape_decode_result.rows()
        );
    }
    if stage_debug || conv_telemetry_debug {
        log_decoder_conv_telemetry("shape_decoder", &shape_conv_telemetry);
        log_decoder_op_telemetry("shape_decoder", &shape_op_telemetry);
        log_neighbor_build_stats("shape_decoder");
    }

    let tex_decode_result = if needs_tex_decode {
        let Some(tex_decoder) = tex_decoder else {
            return Err("runtime decode rust o_voxel PBR requires tex runtime decoder".to_string());
        };
        reset_decoder_conv_telemetry();
        reset_decoder_op_telemetry();
        reset_neighbor_build_stats();
        let tex_decode_start = Instant::now();
        trellis_stage_log!(
            "burn_trellis: stage decode.tex_decoder begin (rows={} guides={})",
            count,
            shape_decode_result.subdivisions.len()
        );
        let tex_coords_host = if !tex.coords.is_empty() {
            Some(&tex.coords[..count])
        } else {
            None
        };
        #[cfg(feature = "runtime-model-wgpu")]
        let tex_decode_result = if using_device_decode_inputs {
            let tex_coords_wgpu_for_decode = if let Some(coords_t) = tex_coords_wgpu.as_ref() {
                let [rows, cols] = coords_t.dims();
                if cols != 4 {
                    return Err(format!(
                        "runtime decode tex coord tensor must have 4 columns, got {}",
                        cols
                    ));
                }
                if rows < count {
                    return Err(format!(
                        "runtime decode tex coord tensor rows {} smaller than requested count {}",
                        rows, count
                    ));
                }
                if rows == count {
                    coords_t.clone()
                } else {
                    coords_t.clone().slice([0..count, 0..4])
                }
            } else {
                return Err(
                "runtime decode canonical wgpu path requires device tex coords; shape-coord fallback is disabled"
                    .to_string(),
            );
            };
            let tex_rows_wgpu_for_decode = if let Some(rows_t) = tex_features_wgpu.as_ref() {
                let [rows, cols] = rows_t.dims();
                if cols != 32 {
                    return Err(format!(
                        "runtime decode tex feature tensor must have 32 columns, got {}",
                        cols
                    ));
                }
                if rows == count {
                    rows_t.clone()
                } else {
                    rows_t.clone().slice([0..count, 0..32])
                }
            } else {
                return Err(
                "runtime decode canonical wgpu path requires device tex rows; host row tensorization fallback is disabled"
                    .to_string(),
            );
            };
            let rows_t = tex_rows_wgpu_for_decode.clone();
            match tex_decoder
                .decode_with_guidance_result_with_tensors_async(
                    tex_coords_wgpu_for_decode.clone(),
                    rows_t,
                    shape_decode_result.subdivisions.as_slice(),
                )
                .await
            {
                Ok(decoded) => decoded,
                Err(err) => {
                    trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
                    return Err(format!("tex runtime decoder failed: {err}"));
                }
            }
        } else {
            match if let Some(coords_host) = tex_coords_host {
                let tex_rows = tex_rows_host.ok_or_else(|| {
                    "runtime decode missing tex host rows for host coord decode path".to_string()
                })?;
                tex_decoder.decode_with_guidance_result(
                    coords_host,
                    tex_rows,
                    shape_decode_result.subdivisions.as_slice(),
                )
            } else {
                return Err("runtime decode missing tex coords for host decode path".to_string());
            } {
                Ok(decoded) => decoded,
                Err(err) => {
                    trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
                    return Err(format!("tex runtime decoder failed: {err}"));
                }
            }
        };
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let tex_decode_result = match if let Some(coords_host) = tex_coords_host {
            let tex_rows = tex_rows_host.ok_or_else(|| {
                "runtime decode missing tex host rows for host coord decode path".to_string()
            })?;
            tex_decoder.decode_with_guidance_result(
                coords_host,
                tex_rows,
                shape_decode_result.subdivisions.as_slice(),
            )
        } else {
            return Err(
                "runtime decode missing tex coords and crate was built without runtime-model-wgpu"
                    .to_string(),
            );
        } {
            Ok(decoded) => decoded,
            Err(err) => {
                trellis_stage_log!("burn_trellis: stage decode.tex_decoder error ({err})");
                return Err(format!("tex runtime decoder failed: {err}"));
            }
        };
        #[cfg(feature = "runtime-model-wgpu")]
        runtime_decode_stage_boundary_sync(
            "tex_decoder",
            using_device_decode_inputs && decode_stage_fenced,
        )?;
        let tex_decoder_ms = tex_decode_start.elapsed().as_secs_f64() * 1000.0;
        trellis_stage_log!(
            "burn_trellis: stage decode.tex_decoder complete ({tex_decoder_ms:.2} ms, coords={})",
            tex_decode_result.rows()
        );
        let tex_conv_telemetry = decoder_conv_telemetry();
        let tex_op_telemetry = decoder_op_telemetry();
        if stage_debug {
            trellis_stage_log!(
                "burn_trellis: decode runtime tex-decoder complete ({:.2} ms, coords={})",
                tex_decoder_ms,
                tex_decode_result.rows()
            );
        }
        if stage_debug || conv_telemetry_debug {
            log_decoder_conv_telemetry("tex_decoder", &tex_conv_telemetry);
            log_decoder_op_telemetry("tex_decoder", &tex_op_telemetry);
            log_neighbor_build_stats("tex_decoder");
        }
        Some((
            tex_decode_result,
            tex_decoder_ms,
            tex_conv_telemetry,
            tex_op_telemetry,
        ))
    } else {
        trellis_stage_log!(
            "burn_trellis: stage decode.tex_decoder skipped (decode_output_mode={})",
            decode_output_mode.as_str()
        );
        None
    };
    let (tex_decode_result, tex_decoder_ms, tex_conv_telemetry, _tex_op_telemetry) =
        match tex_decode_result {
            Some((decode_result, decoder_ms, conv_telemetry, op_telemetry)) => (
                Some(decode_result),
                decoder_ms,
                conv_telemetry,
                op_telemetry,
            ),
            None => (
                None,
                0.0,
                DecoderConvTelemetry::default(),
                DecoderOpTelemetry::default(),
            ),
        };

    if final_resolution == 0 {
        return Err(format!(
            "runtime decode received invalid final resolution for pipeline '{pipeline_type}'"
        ));
    }
    let output_materialize_start = Instant::now();
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_decoded_coords_host = shape_decode_result
        .coords_host_async("runtime decode shape coord stage-boundary materialization")
        .await
        .map_err(|err| format!("shape runtime decoder coord readback failed: {err}"))?;
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let shape_decoded_coords_host = shape_decode_result
        .coords_host("runtime decode shape coord stage-boundary materialization")
        .map_err(|err| format!("shape runtime decoder coord readback failed: {err}"))?;
    #[cfg(feature = "runtime-model-wgpu")]
    let shape_decoded_feats_host = shape_decode_result
        .feats_host_async("runtime decode shape feat stage-boundary materialization")
        .await
        .map_err(|err| format!("shape runtime decoder feat readback failed: {err}"))?;
    #[cfg(not(feature = "runtime-model-wgpu"))]
    let shape_decoded_feats_host = shape_decode_result
        .feats_host("runtime decode shape feat stage-boundary materialization")
        .map_err(|err| format!("shape runtime decoder feat readback failed: {err}"))?;
    let shape_decoded = decode_fdg_outputs_from_host(
        shape_decoded_coords_host,
        shape_decoded_feats_host.as_slice(),
        shape_decode_result.out_channels,
        shape_decode_result.subdivisions.as_slice(),
        shape_decoder.voxel_margin(),
    )
    .map_err(|err| format!("shape runtime decoder output decode failed: {err}"))?;
    let tex_attrs = if let Some(tex_decode_result) = tex_decode_result.as_ref() {
        let tex_decoded_rows = tex_decode_result.rows();
        if tex_decoded_rows != shape_decoded.coords.len() {
            return Err(format!(
                "tex runtime decoder row mismatch: expected_rows={} actual_rows={}",
                shape_decoded.coords.len(),
                tex_decoded_rows
            ));
        }
        #[cfg(feature = "runtime-model-wgpu")]
        let tex_decoded_feats_host = tex_decode_result
            .feats_host_async("runtime decode tex feat stage-boundary materialization")
            .await
            .map_err(|err| format!("tex runtime decoder feat readback failed: {err}"))?;
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let tex_decoded_feats_host = tex_decode_result
            .feats_host("runtime decode tex feat stage-boundary materialization")
            .map_err(|err| format!("tex runtime decoder feat readback failed: {err}"))?;
        decode_tex_attrs_from_host(
            tex_decoded_feats_host.as_slice(),
            tex_decode_result.out_channels,
            Some(shape_decoded.coords.len()),
        )
        .map_err(|err| format!("tex runtime decoder output decode failed: {err}"))?
    } else {
        Vec::new()
    };
    let output_materialize_ms = output_materialize_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!(
        "burn_trellis: stage decode.output_materialize complete ({output_materialize_ms:.2} ms, rows={})",
        shape_decoded.coords.len()
    );
    let shape_subdivisions = shape_decoded.subdivisions;
    let coords = shape_decoded.coords;
    let shape_vertices = shape_decoded.vertices;
    let shape_intersected = shape_decoded.intersected;
    let _shape_intersection_logits = shape_decoded.intersection_logits;
    let shape_quad_lerp = shape_decoded.quad_lerp;
    trellis_stage_log!(
        "burn_trellis: stage decode.attr_merge begin (rows={})",
        coords.len()
    );
    let attr_merge_start = Instant::now();
    let voxel_attrs = if needs_tex_decode {
        merge_voxel_attrs_for_decode(
            coords.as_slice(),
            coords.as_slice(),
            tex_attrs.as_slice(),
            parity_strict,
        )?
    } else {
        Vec::new()
    };
    let attr_merge_ms = attr_merge_start.elapsed().as_secs_f64() * 1000.0;
    trellis_stage_log!("burn_trellis: stage decode.attr_merge complete ({attr_merge_ms:.2} ms)");
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime attr merge complete ({:.2} ms)",
            attr_merge_ms
        );
    }

    let grid_size = [
        final_resolution as u32,
        final_resolution as u32,
        final_resolution as u32,
    ];
    trellis_stage_log!(
        "burn_trellis: stage decode.mesh_extract begin (rows={} final_res={})",
        coords.len(),
        final_resolution
    );
    let mesh_start = Instant::now();
    let mut vertices;
    let mut faces;
    (vertices, faces) = flexible_dual_grid_to_mesh(
        &coords,
        shape_vertices.as_slice(),
        shape_intersected.as_slice(),
        Some(shape_quad_lerp.as_slice()),
        grid_size,
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
    );
    let mesh_ms = mesh_start.elapsed().as_secs_f64() * 1000.0;
    let (
        sanitized_vertices,
        sanitized_faces,
        dropped_vertices,
        dropped_invalid_faces,
        dropped_degenerate_faces,
    ) = sanitize_mesh_geometry(vertices, faces);
    vertices = sanitized_vertices;
    faces = sanitized_faces;
    if dropped_vertices > 0 || dropped_invalid_faces > 0 || dropped_degenerate_faces > 0 {
        trellis_stage_log!(
            "burn_trellis: decode runtime mesh sanitized (dropped_vertices={} dropped_invalid_faces={} dropped_degenerate_faces={})",
            dropped_vertices,
            dropped_invalid_faces,
            dropped_degenerate_faces
        );
    }
    trellis_stage_log!(
        "burn_trellis: stage decode.mesh_extract complete ({mesh_ms:.2} ms, vertices={}, faces={})",
        vertices.len(),
        faces.len()
    );
    if stage_debug {
        trellis_stage_log!(
            "burn_trellis: decode runtime mesh complete ({:.2} ms, vertices={}, faces={})",
            mesh_ms,
            vertices.len(),
            faces.len()
        );
    }
    if vertices.is_empty() || faces.is_empty() {
        return Err("runtime decode produced empty mesh".to_string());
    }
    let projection_vertices = vertices.clone();
    let projection_faces = faces.clone();
    let mut use_projection_source = false;
    #[cfg(not(target_arch = "wasm32"))]
    let mut projection_bvh_for_pbr = None;
    let mut remesh_bvh_ms = 0.0f64;
    let mut remesh_refine_ms = 0.0f64;
    let mut remesh_dc_ms = 0.0f64;
    let mut remesh_cleanup_ms = 0.0f64;
    let mut remesh_total_ms = 0.0f64;
    #[cfg(not(target_arch = "wasm32"))]
    {
        if decode_output_mode.needs_native_mesh_postprocess() {
            let remesh_start = Instant::now();
            let before_remesh_faces = faces.len();
            let projection_bvh_start = Instant::now();
            let projection_bvh = build_projection_bvh_for_pbr(PbrProjectionSource {
                vertices: projection_vertices.as_slice(),
                faces: projection_faces.as_slice(),
            })
            .map_err(|err| format!("runtime decode rust o_voxel mesh remesh failed: {err}"))?;
            remesh_bvh_ms = projection_bvh_start.elapsed().as_secs_f64() * 1000.0;
            let remesh_output = remesh_narrow_band_simple_dc_with_projection_bvh(
                &projection_bvh,
                final_resolution as u32,
                NATIVE_PBR_OVOXEL_REMESH_BAND,
                None,
            )
            .map_err(|err| format!("runtime decode rust o_voxel mesh remesh failed: {err}"))?;
            remesh_refine_ms = remesh_output.refine_ms;
            remesh_dc_ms = remesh_output.dc_ms;
            projection_bvh_for_pbr = Some(projection_bvh);
            vertices = remesh_output.vertices;
            faces = remesh_output.faces;
            let remesh_cleanup_start = Instant::now();
            let remesh_cleanup = cleanup_mesh_topology(&mut vertices, &mut faces);
            remesh_cleanup_ms = remesh_cleanup_start.elapsed().as_secs_f64() * 1000.0;
            if remesh_cleanup.duplicate_faces_removed > 0
                || remesh_cleanup.nonmanifold_faces_removed > 0
                || remesh_cleanup.small_component_faces_removed > 0
                || remesh_cleanup.hole_faces_added > 0
            {
                trellis_stage_log!(
                    "burn_trellis: runtime decode rust_ovoxel mesh remesh cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} small_component_faces_removed={} hole_faces_added={} vertices={} faces={})",
                    remesh_cleanup.duplicate_faces_removed,
                    remesh_cleanup.nonmanifold_faces_removed,
                    remesh_cleanup.small_component_faces_removed,
                    remesh_cleanup.hole_faces_added,
                    vertices.len(),
                    faces.len()
                );
            }
            remesh_total_ms = remesh_start.elapsed().as_secs_f64() * 1000.0;
            trellis_stage_log!(
                "burn_trellis: runtime decode rust_ovoxel mesh remesh complete ({:.2} ms, bvh={:.2} ms refine={:.2} ms dc={:.2} ms cleanup={:.2} ms active_voxels={} grid_vertices={} from_faces={} to_faces={} vertices={})",
                remesh_total_ms,
                remesh_bvh_ms,
                remesh_refine_ms,
                remesh_dc_ms,
                remesh_cleanup_ms,
                remesh_output.active_voxels,
                remesh_output.grid_vertices,
                before_remesh_faces,
                faces.len(),
                vertices.len()
            );
            if vertices.is_empty() || faces.is_empty() {
                return Err("runtime decode rust o_voxel mesh remesh produced empty mesh".to_string());
            }
            use_projection_source = true;
        }
    }
    #[cfg(target_arch = "wasm32")]
    if decode_output_mode.needs_native_mesh_postprocess() {
        return Err(format!(
            "runtime decode {} requires Rust o_voxel mesh remesh/postprocess, which is not available on wasm yet; refusing raw FDG output",
            decode_output_mode.as_str()
        ));
    }
    let mut pre_pbr_decimate_ms = 0.0f64;
    if let Some(target_faces) = target_faces.filter(|limit| *limit > 0) {
        if faces.len() > target_faces {
            let before_faces = faces.len();
            let decimate_start = Instant::now();
            decimate_mesh_for_face_budget(&mut vertices, &mut faces, target_faces)?;
            pre_pbr_decimate_ms = decimate_start.elapsed().as_secs_f64() * 1000.0;
            use_projection_source = true;
            trellis_stage_log!(
                "burn_trellis: runtime decode pre-pbr decimation complete ({:.2} ms, target_faces={} from_faces={} to_faces={})",
                pre_pbr_decimate_ms,
                target_faces,
                before_faces,
                faces.len()
            );
            if vertices.is_empty() || faces.is_empty() {
                return Err("runtime decode pre-pbr decimation produced empty mesh".to_string());
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let decimate_cleanup = cleanup_mesh_topology(&mut vertices, &mut faces);
                if decimate_cleanup.duplicate_faces_removed > 0
                    || decimate_cleanup.nonmanifold_faces_removed > 0
                    || decimate_cleanup.small_component_faces_removed > 0
                    || decimate_cleanup.hole_faces_added > 0
                {
                    trellis_stage_log!(
                        "burn_trellis: runtime decode pre-pbr decimation cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} small_component_faces_removed={} hole_faces_added={} vertices={} faces={})",
                        decimate_cleanup.duplicate_faces_removed,
                        decimate_cleanup.nonmanifold_faces_removed,
                        decimate_cleanup.small_component_faces_removed,
                        decimate_cleanup.hole_faces_added,
                        vertices.len(),
                        faces.len()
                    );
                }
                if vertices.is_empty() || faces.is_empty() {
                    return Err(
                        "runtime decode pre-pbr decimation cleanup produced empty mesh".to_string(),
                    );
                }
            }
        }
    }
    let (pbr_mesh, pbr_textures, pbr_debug, pbr_ms) = if decode_output_mode.needs_native_pbr() {
        trellis_stage_log!("burn_trellis: stage decode.pbr begin");
        let pbr_start = Instant::now();
        #[cfg(feature = "runtime-model-wgpu")]
        // Canonical device decode should keep decode/PBR sampling on-device whenever
        // tensor-native decode inputs are active; CPU PBR sampling remains available
        // only for explicit host decode mode.
        let prefer_wgpu_sampling = using_device_decode_inputs;
        #[cfg(not(feature = "runtime-model-wgpu"))]
        let prefer_wgpu_sampling = false;
        #[cfg(not(target_arch = "wasm32"))]
        let projection_bvh_override = if use_projection_source {
            projection_bvh_for_pbr.as_ref()
        } else {
            None
        };
        #[cfg(target_arch = "wasm32")]
        let projection_bvh_override = None;
        #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
        let (mut pbr_mesh, pbr_textures, pbr_debug) =
            bake_pbr_from_voxels_with_options_and_projection_bvh(
                vertices.as_slice(),
                faces.as_slice(),
                use_projection_source.then_some(PbrProjectionSource {
                    vertices: projection_vertices.as_slice(),
                    faces: projection_faces.as_slice(),
                }),
                projection_bvh_override,
                coords.as_slice(),
                voxel_attrs.as_slice(),
                final_resolution as u32,
                pbr_texture_size,
                capture_debug_artifacts,
                prefer_wgpu_sampling,
            )
            .map_err(|err| format!("runtime decode pbr bake failed: {err}"))?;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pbr_cleanup = cleanup_pbr_bake_mesh_topology(&mut pbr_mesh);
            if pbr_cleanup.duplicate_faces_removed > 0 || pbr_cleanup.nonmanifold_faces_removed > 0
            {
                trellis_stage_log!(
                    "burn_trellis: runtime decode pbr output cleanup complete (duplicate_faces_removed={} nonmanifold_faces_removed={} vertices={} faces={} uvs={})",
                    pbr_cleanup.duplicate_faces_removed,
                    pbr_cleanup.nonmanifold_faces_removed,
                    pbr_mesh.vertices.len(),
                    pbr_mesh.faces.len(),
                    pbr_mesh.uvs.len()
                );
            }
        }
        let pbr_ms = pbr_start.elapsed().as_secs_f64() * 1000.0;
        trellis_stage_log!("burn_trellis: stage decode.pbr complete ({pbr_ms:.2} ms)");
        if stage_debug {
            trellis_stage_log!("burn_trellis: decode runtime pbr complete ({pbr_ms:.2} ms)");
        }
        (pbr_mesh, pbr_textures, pbr_debug, pbr_ms)
    } else {
        trellis_stage_log!(
            "burn_trellis: stage decode.pbr skipped (decode_output_mode={})",
            decode_output_mode.as_str()
        );
        (
            PbrBakeMesh {
                vertices,
                faces,
                uvs: Vec::new(),
                normals: Vec::new(),
            },
            None,
            None,
            0.0,
        )
    };
    if parity_strict
        && decode_output_mode.needs_native_pbr()
        && (pbr_textures.is_none() || pbr_mesh.uvs.len() != pbr_mesh.vertices.len())
    {
        return Err(format!(
            "parity strict mode: runtime decode pbr mismatch (textures_present={} uvs={} vertices={})",
            pbr_textures.is_some(),
            pbr_mesh.uvs.len(),
            pbr_mesh.vertices.len()
        ));
    }
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mut mesh = Mesh {
        vertices: pbr_mesh.vertices,
        faces: pbr_mesh.faces,
        uvs: pbr_mesh.uvs,
        normals: pbr_mesh.normals,
        material,
        pbr_textures,
    };
    if decode_output_mode.needs_native_mesh_postprocess() {
        apply_native_pbr_ovoxel_remesh_domain_scale(&mut mesh, final_resolution as u32);
    }

    let shape_subs = if capture_debug_artifacts {
        shape_subdivisions
            .iter()
            .map(runtime_subdivision_to_sample)
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    let tex_spatial = spatial_shape_from_sparse_coords(coords.as_slice());

    Ok(DecodedLatentOutput {
        source: DecodeStageSource::Runtime,
        mesh,
        shape_subs,
        tex_voxels: DecodeTexVoxelSample {
            coords,
            feats: voxel_attrs,
            spatial_shape: tex_spatial,
        },
        pbr: pbr_debug,
        timings: DecodeRuntimeTimings {
            stage_fenced: decode_stage_fenced,
            shape_decoder_ms,
            tex_decoder_ms,
            output_materialize_ms,
            attr_merge_ms,
            mesh_ms,
            remesh_bvh_ms,
            remesh_refine_ms,
            remesh_dc_ms,
            remesh_cleanup_ms,
            remesh_total_ms,
            pre_pbr_decimate_ms,
            pbr_ms,
            shape_conv_calls: shape_conv_telemetry.conv_calls,
            tex_conv_calls: tex_conv_telemetry.conv_calls,
            shape_wgpu_dispatches: shape_conv_telemetry.dispatches,
            tex_wgpu_dispatches: tex_conv_telemetry.dispatches,
            shape_wgpu_chunked_calls: shape_conv_telemetry.chunked_calls,
            tex_wgpu_chunked_calls: tex_conv_telemetry.chunked_calls,
            shape_wgpu_input_bytes: shape_conv_telemetry.input_bytes,
            tex_wgpu_input_bytes: tex_conv_telemetry.input_bytes,
            shape_wgpu_output_bytes: shape_conv_telemetry.output_bytes,
            tex_wgpu_output_bytes: tex_conv_telemetry.output_bytes,
            shape_wgpu_max_chunk_rows: shape_conv_telemetry.max_chunk_rows,
            tex_wgpu_max_chunk_rows: tex_conv_telemetry.max_chunk_rows,
        },
    })
}

#[cfg(feature = "runtime-model-wgpu")]
fn runtime_decode_uses_device_inputs(
    shape_coords_wgpu: bool,
    tex_coords_wgpu: bool,
    shape_features_wgpu: bool,
    tex_features_wgpu: bool,
) -> bool {
    shape_coords_wgpu || tex_coords_wgpu || shape_features_wgpu || tex_features_wgpu
}

#[cfg(feature = "runtime-model-wgpu")]
fn runtime_decode_stage_boundary_sync(stage: &str, enabled: bool) -> Result<(), String> {
    if !enabled {
        return Ok(());
    }
    // WGPU dispatch is asynchronous; fence here so per-stage decode timing includes
    // real GPU execution instead of spilling completion into later decode stages.
    <SparseFlowWgpuBackend as Backend>::sync(&WgpuDevice::default())
        .map_err(|err| format!("runtime decode {stage} device sync failed: {err}"))
}

#[cfg(all(test, feature = "runtime-model"))]
mod runtime_decode_tests {
    use super::PbrBakeMesh;
    use super::apply_native_pbr_ovoxel_remesh_domain_scale;
    use super::cleanup_mesh_topology;
    use super::cleanup_pbr_bake_mesh_topology;
    #[cfg(not(target_arch = "wasm32"))]
    use super::decimate_mesh_for_face_budget;
    #[cfg(not(target_arch = "wasm32"))]
    use super::{mesh_find, mesh_union};
    use super::sanitize_mesh_geometry;
    use crate::mesh::Mesh;

    #[cfg(feature = "runtime-model-wgpu")]
    use super::runtime_decode_uses_device_inputs;

    #[test]
    fn sanitize_mesh_geometry_removes_non_finite_vertices_and_reindexes_faces() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [f32::NAN, 0.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [1, 3, 2]];
        let (vertices, faces, dropped_vertices, dropped_invalid_faces, dropped_degenerate_faces) =
            sanitize_mesh_geometry(vertices, faces);
        assert_eq!(dropped_vertices, 1);
        assert_eq!(dropped_invalid_faces, 1);
        assert_eq!(dropped_degenerate_faces, 0);
        assert_eq!(vertices.len(), 3);
        assert_eq!(faces, vec![[0, 1, 2]]);
    }

    #[test]
    fn sanitize_mesh_geometry_drops_degenerate_faces() {
        let vertices = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let faces = vec![[0, 1, 1], [0, 1, 2]];
        let (_vertices, faces, dropped_vertices, dropped_invalid_faces, dropped_degenerate_faces) =
            sanitize_mesh_geometry(vertices, faces);
        assert_eq!(dropped_vertices, 0);
        assert_eq!(dropped_invalid_faces, 0);
        assert_eq!(dropped_degenerate_faces, 1);
        assert_eq!(faces, vec![[0, 1, 2]]);
    }

    #[test]
    fn cleanup_mesh_topology_removes_duplicates_and_tiny_components() {
        let mut vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.001, 0.0, 0.0],
            [2.0, 0.001, 0.0],
        ];
        let mut faces = vec![[0, 1, 2], [2, 1, 0], [3, 4, 5]];

        let stats = cleanup_mesh_topology(&mut vertices, &mut faces);

        assert_eq!(stats.duplicate_faces_removed, 1);
        assert_eq!(stats.small_component_faces_removed, 1);
        assert_eq!(stats.hole_faces_added, 0);
        assert_eq!(faces.len(), 1);
        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn cleanup_mesh_topology_removes_relative_area_fragments() {
        let mut vertices = vec![
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [20.0, 0.0, 0.0],
            [20.02, 0.0, 0.0],
            [20.0, 0.02, 0.0],
        ];
        let mut faces = vec![[0, 1, 2], [3, 4, 5]];

        let stats = cleanup_mesh_topology(&mut vertices, &mut faces);

        assert_eq!(stats.small_component_faces_removed, 1);
        assert_eq!(faces.len(), 1);
        assert_eq!(vertices.len(), 3);
    }

    #[test]
    fn cleanup_mesh_topology_removes_nonmanifold_edge_faces() {
        let mut vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut faces = vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]];

        let stats = cleanup_mesh_topology(&mut vertices, &mut faces);

        assert_eq!(stats.duplicate_faces_removed, 0);
        assert_eq!(stats.nonmanifold_faces_removed, 1);
        assert_eq!(faces.len(), 2);
        let mut edge_counts = std::collections::HashMap::<(u32, u32), usize>::new();
        for face in &faces {
            for [a, b] in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
                let key = if a <= b { (a, b) } else { (b, a) };
                *edge_counts.entry(key).or_default() += 1;
            }
        }
        assert!(edge_counts.values().all(|count| *count <= 2));
    }

    #[test]
    fn cleanup_pbr_bake_mesh_topology_preserves_uv_alignment() {
        let mut mesh = PbrBakeMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            faces: vec![[0, 1, 2], [1, 0, 3], [0, 1, 4]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.5], [0.5, 0.5]],
            normals: Vec::new(),
        };

        let stats = cleanup_pbr_bake_mesh_topology(&mut mesh);

        assert_eq!(stats.nonmanifold_faces_removed, 1);
        assert_eq!(mesh.faces.len(), 2);
        assert_eq!(mesh.uvs.len(), mesh.vertices.len());
        assert!(
            mesh.faces
                .iter()
                .flatten()
                .all(|idx| (*idx as usize) < mesh.vertices.len())
        );
    }

    #[test]
    fn cleanup_mesh_topology_fills_small_boundary_loop() {
        let mut vertices = vec![
            [0.0, 0.0, 0.0],
            [0.005, 0.0, 0.0],
            [0.0, 0.005, 0.0],
            [0.0, 0.0, 0.005],
        ];
        let mut faces = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];

        let stats = cleanup_mesh_topology(&mut vertices, &mut faces);

        assert_eq!(stats.duplicate_faces_removed, 0);
        assert_eq!(stats.small_component_faces_removed, 0);
        assert_eq!(stats.hole_faces_added, 0);
        assert_eq!(faces.len(), 4);
        assert_eq!(vertices.len(), 4);

        faces.pop();
        let stats = cleanup_mesh_topology(&mut vertices, &mut faces);
        assert!(stats.hole_faces_added >= 3);
        assert!(faces.len() >= 4);
    }

    #[test]
    fn native_pbr_ovoxel_remesh_domain_scale_matches_upstream_band_formula() {
        let mut mesh = Mesh::new(vec![[1.0, -2.0, 0.5]], vec![[0, 0, 0]]);
        apply_native_pbr_ovoxel_remesh_domain_scale(&mut mesh, 512);
        let expected = (512.0 + 3.0) / 512.0;
        for (actual, base) in mesh.vertices[0].iter().zip([1.0f32, -2.0, 0.5]) {
            assert!((*actual - base * expected).abs() < 1.0e-6);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_decode_pre_pbr_decimation_respects_face_budget() {
        let side = 32usize;
        let mut vertices = Vec::with_capacity((side + 1) * (side + 1));
        for y in 0..=side {
            for x in 0..=side {
                vertices.push([x as f32, y as f32, 0.0]);
            }
        }
        let idx = |x: usize, y: usize| -> u32 { (y * (side + 1) + x) as u32 };
        let mut faces = Vec::with_capacity(side * side * 2);
        for y in 0..side {
            for x in 0..side {
                let i0 = idx(x, y);
                let i1 = idx(x + 1, y);
                let i2 = idx(x, y + 1);
                let i3 = idx(x + 1, y + 1);
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }
        let original_faces = faces.len();
        decimate_mesh_for_face_budget(&mut vertices, &mut faces, 200)
            .expect("runtime decode pre-pbr decimation should succeed");
        assert!(faces.len() <= 200, "faces={} > 200", faces.len());
        assert!(!faces.is_empty());
        assert!(faces.len() < original_faces);
        assert!(!vertices.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_decode_pre_pbr_decimation_handles_extreme_reduction() {
        let side = 64usize;
        let mut vertices = Vec::with_capacity((side + 1) * (side + 1));
        for y in 0..=side {
            for x in 0..=side {
                vertices.push([x as f32, y as f32, (x ^ y) as f32 * 0.001]);
            }
        }
        let idx = |x: usize, y: usize| -> u32 { (y * (side + 1) + x) as u32 };
        let mut faces = Vec::with_capacity(side * side * 2);
        for y in 0..side {
            for x in 0..side {
                let i0 = idx(x, y);
                let i1 = idx(x + 1, y);
                let i2 = idx(x, y + 1);
                let i3 = idx(x + 1, y + 1);
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }
        decimate_mesh_for_face_budget(&mut vertices, &mut faces, 20)
            .expect("extreme runtime decimation should succeed");
        assert!(faces.len() <= 20, "faces={} > 20", faces.len());
        assert!(!faces.is_empty());
        assert!(!vertices.is_empty());
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn runtime_decode_pre_pbr_decimation_preserves_topology_for_subpercent_budget() {
        let side = 128usize;
        let mut vertices = Vec::with_capacity((side + 1) * (side + 1));
        for y in 0..=side {
            for x in 0..=side {
                let xf = x as f32 / side as f32;
                let yf = y as f32 / side as f32;
                vertices.push([
                    xf,
                    yf,
                    0.025 * (xf * std::f32::consts::TAU).sin()
                        * (yf * std::f32::consts::TAU).cos(),
                ]);
            }
        }
        let idx = |x: usize, y: usize| -> u32 { (y * (side + 1) + x) as u32 };
        let mut faces = Vec::with_capacity(side * side * 2);
        for y in 0..side {
            for x in 0..side {
                let i0 = idx(x, y);
                let i1 = idx(x + 1, y);
                let i2 = idx(x, y + 1);
                let i3 = idx(x + 1, y + 1);
                faces.push([i0, i1, i3]);
                faces.push([i0, i3, i2]);
            }
        }

        decimate_mesh_for_face_budget(&mut vertices, &mut faces, 200)
            .expect("sub-percent runtime decimation should succeed");
        assert!(faces.len() <= 200, "faces={} > 200", faces.len());
        assert!(!faces.is_empty());
        assert_eq!(
            connected_face_component_count(&faces),
            1,
            "decimation should preserve one connected surface"
        );
        let before_cleanup_faces = faces.len();
        let cleanup = cleanup_mesh_topology(&mut vertices, &mut faces);
        assert_eq!(cleanup.nonmanifold_faces_removed, 0);
        assert!(
            faces.len() * 100 >= before_cleanup_faces * 95,
            "cleanup should not destroy topology after decimation: before={} after={} stats={cleanup:?}",
            before_cleanup_faces,
            faces.len()
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn connected_face_component_count(faces: &[[u32; 3]]) -> usize {
        if faces.is_empty() {
            return 0;
        }
        let mut parent = (0..faces.len()).collect::<Vec<_>>();
        let mut edge_owner = std::collections::HashMap::<(u32, u32), usize>::new();
        for (face_idx, face) in faces.iter().copied().enumerate() {
            for [a, b] in [[face[0], face[1]], [face[1], face[2]], [face[2], face[0]]] {
                let key = if a <= b { (a, b) } else { (b, a) };
                if let Some(prev_idx) = edge_owner.insert(key, face_idx) {
                    mesh_union(parent.as_mut_slice(), prev_idx, face_idx);
                }
            }
        }
        let mut roots = std::collections::HashSet::<usize>::new();
        for face_idx in 0..faces.len() {
            roots.insert(mesh_find(parent.as_mut_slice(), face_idx));
        }
        roots.len()
    }

    #[cfg(feature = "runtime-model-wgpu")]
    #[test]
    fn runtime_decode_device_gate_allows_host_only_inputs() {
        assert!(!runtime_decode_uses_device_inputs(
            false, false, false, false
        ));
        assert!(runtime_decode_uses_device_inputs(true, false, false, false));
        assert!(runtime_decode_uses_device_inputs(false, true, false, false));
        assert!(runtime_decode_uses_device_inputs(false, false, true, false));
        assert!(runtime_decode_uses_device_inputs(false, false, false, true));
    }
}

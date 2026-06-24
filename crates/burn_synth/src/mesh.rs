use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub material: Option<MeshMaterial>,
    pub pbr_textures: Option<MeshPbrTextures>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshMaterial {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshPbrTextures {
    pub base_color: MeshTexture,
    pub metallic_roughness: MeshTexture,
    pub normal: Option<MeshTexture>,
    pub emissive: Option<MeshTexture>,
    pub occlusion: Option<MeshTexture>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshStats {
    pub vertices: usize,
    pub faces: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshQualityMetrics {
    pub vertices: usize,
    pub faces: usize,
    pub uvs: usize,
    pub invalid_face_indices: usize,
    pub degenerate_faces: usize,
    pub degenerate_face_ratio: f32,
    pub raw_connectivity: MeshConnectivityMetrics,
    pub position_welded_connectivity: MeshConnectivityMetrics,
    pub pbr_textures: MeshPbrTextureMetrics,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshConnectivityMetrics {
    pub unique_vertices: usize,
    pub unique_edges: usize,
    pub boundary_edges: usize,
    pub boundary_edge_ratio: f32,
    pub non_manifold_edges: usize,
    pub connected_components: usize,
    pub largest_component_faces: usize,
    pub largest_component_face_fraction: f32,
    pub tiny_components_le_16_faces: usize,
    pub tiny_component_faces_le_16: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MeshPbrTextureMetrics {
    pub has_pbr_textures: bool,
    pub base_color_width: Option<u32>,
    pub base_color_height: Option<u32>,
    pub base_color_alpha_coverage: Option<f32>,
    pub base_color_luma_stddev: Option<f32>,
}

pub trait MeshLike {
    fn vertices(&self) -> &[[f32; 3]];
    fn faces(&self) -> &[[u32; 3]];
}

impl MeshLike for Mesh {
    fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}

pub fn mesh_stats<M: MeshLike>(mesh: &M) -> MeshStats {
    MeshStats {
        vertices: mesh.vertices().len(),
        faces: mesh.faces().len(),
    }
}

pub fn mesh_quality_metrics(mesh: &Mesh) -> MeshQualityMetrics {
    let invalid_face_indices = mesh
        .faces
        .iter()
        .filter(|face| {
            face.iter()
                .any(|&index| index as usize >= mesh.vertices.len())
        })
        .count();
    let degenerate_faces = mesh
        .faces
        .iter()
        .filter(|face| face_is_degenerate(&mesh.vertices, **face))
        .count();
    let faces = mesh.faces.len();
    MeshQualityMetrics {
        vertices: mesh.vertices.len(),
        faces,
        uvs: mesh.uvs.len(),
        invalid_face_indices,
        degenerate_faces,
        degenerate_face_ratio: degenerate_faces as f32 / faces.max(1) as f32,
        raw_connectivity: connectivity_metrics(&mesh.vertices, &mesh.faces, None),
        position_welded_connectivity: connectivity_metrics(
            &mesh.vertices,
            &mesh.faces,
            Some(1.0e-5),
        ),
        pbr_textures: pbr_texture_metrics(mesh),
    }
}

pub fn mesh_quality_failures(metrics: &MeshQualityMetrics) -> Vec<String> {
    let mut failures = Vec::new();
    if metrics.faces == 0 {
        failures.push("mesh has no faces".to_string());
    }
    if metrics.vertices == 0 {
        failures.push("mesh has no vertices".to_string());
    }
    if metrics.invalid_face_indices > 0 {
        failures.push(format!(
            "mesh has {} faces with out-of-range vertex indices",
            metrics.invalid_face_indices
        ));
    }
    if metrics.degenerate_face_ratio > 0.01 {
        failures.push(format!(
            "mesh degenerate face ratio {:.4} exceeds 0.0100",
            metrics.degenerate_face_ratio
        ));
    }
    let welded = &metrics.position_welded_connectivity;
    if welded.boundary_edge_ratio > 0.05 {
        failures.push(format!(
            "position-welded boundary edge ratio {:.4} exceeds 0.0500",
            welded.boundary_edge_ratio
        ));
    }
    if welded.non_manifold_edges > welded.unique_edges.saturating_div(100).max(64) {
        failures.push(format!(
            "position-welded non-manifold edge count {} is high for {} edges",
            welded.non_manifold_edges, welded.unique_edges
        ));
    }
    if welded.connected_components > 128 && welded.largest_component_face_fraction < 0.85 {
        failures.push(format!(
            "position-welded topology has {} components and only {:.3} of faces in the largest component",
            welded.connected_components, welded.largest_component_face_fraction
        ));
    }
    failures
}

pub fn mesh_bounds<M: MeshLike>(mesh: &M) -> Option<([f32; 3], [f32; 3])> {
    let vertices = mesh.vertices();
    let first = vertices.first()?;
    let mut min = *first;
    let mut max = *first;
    for v in vertices.iter().skip(1) {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    Some((min, max))
}

fn face_is_degenerate(vertices: &[[f32; 3]], face: [u32; 3]) -> bool {
    let [a, b, c] = face;
    if a == b || b == c || a == c {
        return true;
    }
    let Some(a) = vertices.get(a as usize) else {
        return true;
    };
    let Some(b) = vertices.get(b as usize) else {
        return true;
    };
    let Some(c) = vertices.get(c as usize) else {
        return true;
    };
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let area2 = cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2];
    area2 <= 1.0e-20
}

fn connectivity_metrics(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    weld_epsilon: Option<f32>,
) -> MeshConnectivityMetrics {
    if vertices.is_empty() || faces.is_empty() {
        return MeshConnectivityMetrics::default();
    }

    let remap = vertex_remap(vertices, weld_epsilon);
    let unique_vertices = remap
        .iter()
        .copied()
        .max()
        .map(|value| value + 1)
        .unwrap_or_default();
    let mut union_find = UnionFind::new(unique_vertices);
    let mut edge_counts: HashMap<[usize; 2], usize> = HashMap::new();
    let mut valid_faces = Vec::with_capacity(faces.len());

    for face in faces {
        let Some(mapped) = remap_face(&remap, *face) else {
            continue;
        };
        if mapped[0] == mapped[1] || mapped[1] == mapped[2] || mapped[0] == mapped[2] {
            continue;
        }
        for [a, b] in [
            [mapped[0], mapped[1]],
            [mapped[1], mapped[2]],
            [mapped[2], mapped[0]],
        ] {
            let edge = if a < b { [a, b] } else { [b, a] };
            *edge_counts.entry(edge).or_default() += 1;
            union_find.union(a, b);
        }
        valid_faces.push(mapped);
    }

    let mut face_components: HashMap<usize, usize> = HashMap::new();
    for face in &valid_faces {
        let root = union_find.find(face[0]);
        *face_components.entry(root).or_default() += 1;
    }
    let largest_component_faces = face_components.values().copied().max().unwrap_or_default();
    let tiny_components: Vec<usize> = face_components
        .values()
        .copied()
        .filter(|&faces| faces <= 16)
        .collect();
    let boundary_edges = edge_counts.values().filter(|&&count| count == 1).count();
    let non_manifold_edges = edge_counts.values().filter(|&&count| count > 2).count();
    let unique_edges = edge_counts.len();
    let face_count = valid_faces.len().max(1);
    MeshConnectivityMetrics {
        unique_vertices,
        unique_edges,
        boundary_edges,
        boundary_edge_ratio: boundary_edges as f32 / unique_edges.max(1) as f32,
        non_manifold_edges,
        connected_components: face_components.len(),
        largest_component_faces,
        largest_component_face_fraction: largest_component_faces as f32 / face_count as f32,
        tiny_components_le_16_faces: tiny_components.len(),
        tiny_component_faces_le_16: tiny_components.iter().sum(),
    }
}

fn vertex_remap(vertices: &[[f32; 3]], weld_epsilon: Option<f32>) -> Vec<usize> {
    let Some(epsilon) = weld_epsilon else {
        return (0..vertices.len()).collect();
    };
    let inv = 1.0 / epsilon.max(1.0e-12);
    let mut keys = HashMap::<[i64; 3], usize>::new();
    let mut remap = Vec::with_capacity(vertices.len());
    for vertex in vertices {
        let key = [
            (vertex[0] * inv).round() as i64,
            (vertex[1] * inv).round() as i64,
            (vertex[2] * inv).round() as i64,
        ];
        let next = keys.len();
        let index = *keys.entry(key).or_insert(next);
        remap.push(index);
    }
    remap
}

fn remap_face(remap: &[usize], face: [u32; 3]) -> Option<[usize; 3]> {
    Some([
        *remap.get(face[0] as usize)?,
        *remap.get(face[1] as usize)?,
        *remap.get(face[2] as usize)?,
    ])
}

fn pbr_texture_metrics(mesh: &Mesh) -> MeshPbrTextureMetrics {
    let Some(textures) = mesh.pbr_textures.as_ref() else {
        return MeshPbrTextureMetrics::default();
    };
    let base = &textures.base_color;
    let pixels = base.rgba8.chunks_exact(4);
    let mut count = 0usize;
    let mut alpha_count = 0usize;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for pixel in pixels {
        count += 1;
        if pixel[3] > 0 {
            alpha_count += 1;
        }
        let luma = 0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64;
        sum += luma;
        sum_sq += luma * luma;
    }
    let alpha_coverage = if count == 0 {
        None
    } else {
        Some(alpha_count as f32 / count as f32)
    };
    let luma_stddev = if count == 0 {
        None
    } else {
        let mean = sum / count as f64;
        Some((sum_sq / count as f64 - mean * mean).max(0.0).sqrt() as f32)
    };
    MeshPbrTextureMetrics {
        has_pbr_textures: true,
        base_color_width: Some(base.width),
        base_color_height: Some(base.height),
        base_color_alpha_coverage: alpha_coverage,
        base_color_luma_stddev: luma_stddev,
    }
}

#[derive(Clone, Debug)]
struct UnionFind {
    parents: Vec<usize>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parents: (0..len).collect(),
        }
    }

    fn find(&mut self, index: usize) -> usize {
        let parent = self.parents[index];
        if parent == index {
            index
        } else {
            let root = self.find(parent);
            self.parents[index] = root;
            root
        }
    }

    fn union(&mut self, a: usize, b: usize) {
        let a = self.find(a);
        let b = self.find(b);
        if a != b {
            self.parents[b] = a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_quality_allows_uv_seam_split_when_positions_weld_cleanly() {
        let tetra = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [0.5, 0.2887, 0.816],
        ];
        let face_indices = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]];
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for face in face_indices {
            let base = vertices.len() as u32;
            vertices.extend(face.map(|index| tetra[index]));
            faces.push([base, base + 1, base + 2]);
        }
        let mesh = Mesh {
            vertices,
            faces,
            uvs: vec![[0.0, 0.0]; 12],
            material: None,
            pbr_textures: None,
        };

        let metrics = mesh_quality_metrics(&mesh);

        assert_eq!(metrics.raw_connectivity.connected_components, 4);
        assert_eq!(metrics.position_welded_connectivity.connected_components, 1);
        assert!(mesh_quality_failures(&metrics).is_empty());
    }

    #[test]
    fn mesh_quality_rejects_real_fragmented_open_topology() {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        for i in 0..150u32 {
            let x = i as f32 * 3.0;
            let base = vertices.len() as u32;
            vertices.extend([[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]]);
            faces.push([base, base + 1, base + 2]);
        }
        let mesh = Mesh {
            vertices,
            faces,
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        };

        let metrics = mesh_quality_metrics(&mesh);
        let failures = mesh_quality_failures(&metrics);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("boundary edge ratio")),
            "{failures:?}"
        );
    }
}

#[cfg(feature = "triposg")]
impl From<burn_tripo::pipeline::mesh::Mesh> for Mesh {
    fn from(value: burn_tripo::pipeline::mesh::Mesh) -> Self {
        Self {
            vertices: value.vertices,
            faces: value.faces,
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        }
    }
}

#[cfg(feature = "triposg")]
impl MeshLike for burn_tripo::pipeline::mesh::Mesh {
    fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}

#[cfg(feature = "trellis")]
impl From<burn_trellis::Mesh> for Mesh {
    fn from(value: burn_trellis::Mesh) -> Self {
        Self {
            vertices: value.vertices,
            faces: value.faces,
            uvs: value.uvs,
            material: value.material.map(|material| MeshMaterial {
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                alpha: material.alpha,
            }),
            pbr_textures: value.pbr_textures.map(|textures| MeshPbrTextures {
                base_color: MeshTexture {
                    width: textures.base_color.width,
                    height: textures.base_color.height,
                    rgba8: textures.base_color.rgba8,
                },
                metallic_roughness: MeshTexture {
                    width: textures.metallic_roughness.width,
                    height: textures.metallic_roughness.height,
                    rgba8: textures.metallic_roughness.rgba8,
                },
                normal: textures.normal.map(|texture| MeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
                emissive: textures.emissive.map(|texture| MeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
                occlusion: textures.occlusion.map(|texture| MeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
            }),
        }
    }
}

#[cfg(feature = "trellis")]
impl MeshLike for burn_trellis::Mesh {
    fn vertices(&self) -> &[[f32; 3]] {
        &self.vertices
    }

    fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }
}

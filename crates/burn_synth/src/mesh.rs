use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub normals: Vec<[f32; 3]>,
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
    pub base_color_luma_mean: Option<f32>,
    pub base_color_luma_stddev: Option<f32>,
    pub base_color_dark_island_fraction: Option<f32>,
    pub base_color_bright_island_fraction: Option<f32>,
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
    let pbr = &metrics.pbr_textures;
    if let (
        Some(alpha_coverage),
        Some(luma_mean),
        Some(luma_stddev),
        Some(dark_fraction),
        Some(bright_fraction),
    ) = (
        pbr.base_color_alpha_coverage,
        pbr.base_color_luma_mean,
        pbr.base_color_luma_stddev,
        pbr.base_color_dark_island_fraction,
        pbr.base_color_bright_island_fraction,
    ) && alpha_coverage > 0.95
        && luma_mean > 120.0
        && luma_stddev > 55.0
        && dark_fraction > 0.24
        && bright_fraction > 0.34
    {
        failures.push(format!(
            "pbr base-color texture has severe light-surface island artifacts (mean={luma_mean:.2} stddev={luma_stddev:.2} dark_fraction={dark_fraction:.3} bright_fraction={bright_fraction:.3})"
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

pub fn compute_vertex_normals(vertices: &[[f32; 3]], faces: &[[u32; 3]]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; vertices.len()];
    let vertex_count = vertices.len() as u32;
    for face in faces {
        let [i0, i1, i2] = *face;
        if i0 >= vertex_count || i1 >= vertex_count || i2 >= vertex_count {
            continue;
        }
        let v0 = vertices[i0 as usize];
        let v1 = vertices[i1 as usize];
        let v2 = vertices[i2 as usize];
        let normal = vec3_cross(vec3_sub(v1, v0), vec3_sub(v2, v0));
        if vec3_len2(normal) <= 1.0e-20 {
            continue;
        }
        for idx in [i0, i1, i2] {
            let entry = &mut normals[idx as usize];
            entry[0] += normal[0];
            entry[1] += normal[1];
            entry[2] += normal[2];
        }
    }
    for normal in &mut normals {
        *normal = normalize_or_up(*normal);
    }
    normals
}

pub fn compute_position_welded_normals(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    weld_epsilon: f32,
    smooth_cos: f32,
) -> Vec<[f32; 3]> {
    let mut normals = compute_vertex_normals(vertices, faces);
    if vertices.is_empty() {
        return normals;
    }

    let inv = 1.0 / weld_epsilon.max(1.0e-12);
    let mut groups = HashMap::<[i64; 3], Vec<usize>>::new();
    for (idx, vertex) in vertices.iter().enumerate() {
        let key = [
            (vertex[0] * inv).round() as i64,
            (vertex[1] * inv).round() as i64,
            (vertex[2] * inv).round() as i64,
        ];
        groups.entry(key).or_default().push(idx);
    }

    for members in groups.values() {
        if members.len() <= 1 {
            continue;
        }

        let mut clusters: Vec<([f32; 3], Vec<usize>)> = Vec::new();
        for &idx in members {
            let normal = normals[idx];
            if let Some((cluster_sum, cluster_members)) = clusters.iter_mut().find(|(sum, _)| {
                let cluster_normal = normalize_or_up(*sum);
                vec3_dot(cluster_normal, normal) >= smooth_cos
            }) {
                cluster_sum[0] += normal[0];
                cluster_sum[1] += normal[1];
                cluster_sum[2] += normal[2];
                cluster_members.push(idx);
            } else {
                clusters.push((normal, vec![idx]));
            }
        }

        for (cluster_sum, cluster_members) in clusters {
            let cluster_normal = normalize_or_up(cluster_sum);
            for idx in cluster_members {
                normals[idx] = cluster_normal;
            }
        }
    }

    normals
}

pub fn align_normals_with_faces(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    normals: &mut [[f32; 3]],
) {
    if normals.len() != vertices.len() {
        return;
    }
    let mut reference = vec![[0.0f32; 3]; vertices.len()];
    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        let a = vertices[i0];
        let b = vertices[i1];
        let c = vertices[i2];
        let face_normal = vec3_cross(vec3_sub(b, a), vec3_sub(c, a));
        for idx in [i0, i1, i2] {
            reference[idx][0] += face_normal[0];
            reference[idx][1] += face_normal[1];
            reference[idx][2] += face_normal[2];
        }
    }
    for (normal, expected) in normals.iter_mut().zip(reference.iter()) {
        if vec3_dot(*normal, *expected) < 0.0 {
            normal[0] = -normal[0];
            normal[1] = -normal[1];
            normal[2] = -normal[2];
        }
    }
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
    let pixels = base.rgba8.as_chunks::<4>().0;
    let mut count = 0usize;
    let mut alpha_count = 0usize;
    let mut lumas = Vec::new();
    for pixel in pixels {
        count += 1;
        if pixel[3] > 0 {
            alpha_count += 1;
            lumas.push(
                0.2126 * pixel[0] as f64 + 0.7152 * pixel[1] as f64 + 0.0722 * pixel[2] as f64,
            );
        }
    }
    let alpha_coverage = if count == 0 {
        None
    } else {
        Some(alpha_count as f32 / count as f32)
    };
    let (luma_mean, luma_stddev, dark_fraction, bright_fraction) = if lumas.is_empty() {
        (None, None, None, None)
    } else {
        let mean = lumas.iter().sum::<f64>() / lumas.len() as f64;
        let sum_sq = lumas.iter().map(|luma| luma * luma).sum::<f64>();
        let stddev = (sum_sq / lumas.len() as f64 - mean * mean).max(0.0).sqrt();
        let dark_cutoff = (mean - 40.0).max(0.0);
        let dark = lumas.iter().filter(|value| **value < dark_cutoff).count();
        let bright = lumas.iter().filter(|value| **value > 200.0).count();
        (
            Some(mean as f32),
            Some(stddev as f32),
            Some(dark as f32 / lumas.len() as f32),
            Some(bright as f32 / lumas.len() as f32),
        )
    };
    MeshPbrTextureMetrics {
        has_pbr_textures: true,
        base_color_width: Some(base.width),
        base_color_height: Some(base.height),
        base_color_alpha_coverage: alpha_coverage,
        base_color_luma_mean: luma_mean,
        base_color_luma_stddev: luma_stddev,
        base_color_dark_island_fraction: dark_fraction,
        base_color_bright_island_fraction: bright_fraction,
    }
}

fn vec3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

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

fn normalize_or_up(normal: [f32; 3]) -> [f32; 3] {
    let len2 = vec3_len2(normal);
    if len2.is_finite() && len2 > 1.0e-12 {
        let inv = len2.sqrt().recip();
        [normal[0] * inv, normal[1] * inv, normal[2] * inv]
    } else {
        [0.0, 1.0, 0.0]
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

    fn solid_texture(width: u32, height: u32, rgba: [u8; 4]) -> MeshTexture {
        let mut rgba8 = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..(width as usize * height as usize) {
            rgba8.extend_from_slice(&rgba);
        }
        MeshTexture {
            width,
            height,
            rgba8,
        }
    }

    fn closed_tetra_mesh_with_base_texture(base_color: MeshTexture) -> Mesh {
        Mesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.866, 0.0],
                [0.5, 0.2887, 0.816],
            ],
            faces: vec![[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]],
            uvs: vec![[0.0, 0.0]; 4],
            normals: Vec::new(),
            material: None,
            pbr_textures: Some(MeshPbrTextures {
                base_color,
                metallic_roughness: solid_texture(4, 4, [0, 128, 0, 255]),
                normal: None,
                emissive: None,
                occlusion: None,
            }),
        }
    }

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
            normals: Vec::new(),
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
            normals: Vec::new(),
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

    #[test]
    fn mesh_quality_rejects_light_surface_texture_islands() {
        let mut rgba8 = Vec::new();
        for idx in 0..16 {
            if idx % 3 == 0 {
                rgba8.extend_from_slice(&[40, 40, 40, 255]);
            } else {
                rgba8.extend_from_slice(&[230, 230, 230, 255]);
            }
        }
        let mesh = closed_tetra_mesh_with_base_texture(MeshTexture {
            width: 4,
            height: 4,
            rgba8,
        });

        let metrics = mesh_quality_metrics(&mesh);
        let failures = mesh_quality_failures(&metrics);

        assert!(
            failures
                .iter()
                .any(|failure| failure.contains("light-surface island artifacts")),
            "{failures:?}"
        );
    }

    #[test]
    fn mesh_quality_allows_dark_uniform_pbr_texture() {
        let mesh = closed_tetra_mesh_with_base_texture(solid_texture(4, 4, [20, 20, 20, 255]));

        let metrics = mesh_quality_metrics(&mesh);
        let failures = mesh_quality_failures(&metrics);

        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(metrics.pbr_textures.base_color_luma_mean, Some(20.0));
        assert_eq!(
            metrics.pbr_textures.base_color_dark_island_fraction,
            Some(0.0)
        );
    }

    #[test]
    fn position_welded_normals_preserve_vertex_count_and_unit_length() {
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];

        let normals = compute_position_welded_normals(&vertices, &faces, 1.0e-5, 0.55);

        assert_eq!(normals.len(), vertices.len());
        for normal in normals {
            let len = vec3_len2(normal).sqrt();
            assert!(
                (len - 1.0).abs() <= 1.0e-5,
                "normal must be unit length: {normal:?}"
            );
        }
    }
}

#[cfg(feature = "triposg")]
impl From<burn_tripo::pipeline::mesh::Mesh> for Mesh {
    fn from(value: burn_tripo::pipeline::mesh::Mesh) -> Self {
        Self {
            vertices: value.vertices,
            faces: value.faces,
            uvs: Vec::new(),
            normals: Vec::new(),
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
            normals: value.normals,
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

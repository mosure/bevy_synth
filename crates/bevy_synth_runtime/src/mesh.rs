use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};

use burn_tripo::pipeline::mesh::Mesh as TripoMesh;

use crate::SynthMesh;

pub fn to_bevy_mesh(mesh: &TripoMesh) -> BevyMesh {
    to_bevy_mesh_with_uvs(mesh, None)
}

pub fn to_bevy_mesh_synth(mesh: &SynthMesh) -> BevyMesh {
    let uvs = if mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty() {
        Some(mesh.uvs.as_slice())
    } else {
        None
    };
    to_bevy_mesh_with_uvs(&mesh.mesh, uvs)
}

fn to_bevy_mesh_with_uvs(mesh: &TripoMesh, uvs_opt: Option<&[[f32; 2]]>) -> BevyMesh {
    let mut bevy_mesh = BevyMesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    let normals = compute_normals(mesh);
    let uvs = uvs_opt
        .map(|uvs| uvs.to_vec())
        .unwrap_or_else(|| vec![[0.0, 0.0]; mesh.vertices.len()]);
    let indices: Vec<u32> = mesh
        .faces
        .iter()
        .flat_map(|face| face.iter().copied())
        .collect();

    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, mesh.vertices.clone());
    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
    bevy_mesh.insert_attribute(BevyMesh::ATTRIBUTE_UV_0, uvs);
    bevy_mesh.insert_indices(Indices::U32(indices));
    let _ = bevy_mesh.generate_tangents();
    bevy_mesh
}

fn compute_normals(mesh: &TripoMesh) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; mesh.vertices.len()];
    for face in &mesh.faces {
        let [i0, i1, i2] = *face;
        let v0 = mesh.vertices[i0 as usize];
        let v1 = mesh.vertices[i1 as usize];
        let v2 = mesh.vertices[i2 as usize];
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &idx in &[i0, i1, i2] {
            let entry = &mut normals[idx as usize];
            entry[0] += n[0];
            entry[1] += n[1];
            entry[2] += n[2];
        }
    }

    for normal in &mut normals {
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if length > 1e-6 {
            normal[0] /= length;
            normal[1] /= length;
            normal[2] /= length;
        } else {
            *normal = [0.0, 1.0, 0.0];
        }
    }

    normals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_synth_mesh() -> SynthMesh {
        SynthMesh {
            mesh: TripoMesh {
                vertices: vec![
                    [-1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, 1.0],
                    [-1.0, 0.0, 1.0],
                ],
                faces: vec![[0, 1, 2], [0, 2, 3]],
            },
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            material: None,
            pbr_textures: None,
        }
    }

    #[test]
    fn bevy_mesh_generation_includes_uvs_and_tangents() {
        let mesh = to_bevy_mesh_synth(&sample_synth_mesh());
        assert!(mesh.contains_attribute(BevyMesh::ATTRIBUTE_UV_0));
        assert!(mesh.contains_attribute(BevyMesh::ATTRIBUTE_TANGENT));
    }
}

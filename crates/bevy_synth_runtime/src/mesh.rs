use bevy_asset::RenderAssetUsages;
use bevy_mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};

use burn_tripo::pipeline::mesh::Mesh as TripoMesh;

use crate::SynthMesh;

pub fn to_bevy_mesh(mesh: &TripoMesh) -> BevyMesh {
    to_bevy_mesh_with_uvs(mesh, None, None, false)
}

pub fn to_bevy_mesh_synth(mesh: &SynthMesh) -> BevyMesh {
    let has_uvs = mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty();
    let uvs = if has_uvs {
        Some(mesh.uvs.as_slice())
    } else {
        None
    };
    let normals = if mesh.normals.len() == mesh.mesh.vertices.len() && !mesh.normals.is_empty() {
        Some(mesh.normals.as_slice())
    } else {
        None
    };
    let has_normal_map = mesh
        .pbr_textures
        .as_ref()
        .and_then(|pbr| pbr.normal.as_ref())
        .is_some();
    to_bevy_mesh_with_uvs(&mesh.mesh, uvs, normals, has_uvs && has_normal_map)
}

fn to_bevy_mesh_with_uvs(
    mesh: &TripoMesh,
    uvs_opt: Option<&[[f32; 2]]>,
    normals_opt: Option<&[[f32; 3]]>,
    generate_tangents: bool,
) -> BevyMesh {
    let mut bevy_mesh = BevyMesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::default(),
    );

    let normals = if let Some(normals) = normals_opt {
        let mut normals = normals.to_vec();
        burn_synth::align_normals_with_faces(
            mesh.vertices.as_slice(),
            mesh.faces.as_slice(),
            normals.as_mut_slice(),
        );
        normals
    } else {
        compute_normals(mesh)
    };
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
    if generate_tangents {
        let _ = bevy_mesh.generate_tangents();
    }
    bevy_mesh
}

pub(crate) fn compute_normals(mesh: &TripoMesh) -> Vec<[f32; 3]> {
    burn_synth::compute_position_welded_normals(&mesh.vertices, &mesh.faces, 1.0e-5, 0.55)
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
            normals: Vec::new(),
            material: None,
            pbr_textures: None,
        }
    }

    fn sample_synth_mesh_with_normal_map() -> SynthMesh {
        SynthMesh {
            pbr_textures: Some(crate::SynthMeshPbrTextures {
                base_color: crate::SynthMeshTexture {
                    width: 2,
                    height: 2,
                    rgba8: vec![255; 16],
                },
                metallic_roughness: crate::SynthMeshTexture {
                    width: 2,
                    height: 2,
                    rgba8: vec![255; 16],
                },
                normal: Some(crate::SynthMeshTexture {
                    width: 2,
                    height: 2,
                    rgba8: vec![255; 16],
                }),
                emissive: None,
                occlusion: None,
            }),
            ..sample_synth_mesh()
        }
    }

    #[test]
    fn bevy_mesh_generation_includes_uvs() {
        let mesh = to_bevy_mesh_synth(&sample_synth_mesh());
        assert!(mesh.contains_attribute(BevyMesh::ATTRIBUTE_UV_0));
        assert!(!mesh.contains_attribute(BevyMesh::ATTRIBUTE_TANGENT));
    }

    #[test]
    fn bevy_mesh_generation_adds_tangents_when_normal_map_present() {
        let mesh = to_bevy_mesh_synth(&sample_synth_mesh_with_normal_map());
        assert!(mesh.contains_attribute(BevyMesh::ATTRIBUTE_TANGENT));
    }
}

use std::fs;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct RenderMesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

impl RenderMesh {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }
}

pub fn load_glb_mesh(path: &Path) -> Result<RenderMesh, String> {
    let bytes =
        fs::read(path).map_err(|err| format!("failed to read GLB {}: {err}", path.display()))?;
    mesh_from_glb_bytes(&bytes)
        .map_err(|err| format!("failed to parse GLB {}: {err}", path.display()))
}

pub fn mesh_from_glb_bytes(bytes: &[u8]) -> Result<RenderMesh, String> {
    let gltf =
        gltf::Gltf::from_slice(bytes).map_err(|err| format!("failed to parse GLB: {err}"))?;
    let blob = gltf
        .blob
        .as_deref()
        .ok_or_else(|| "GLB binary chunk missing".to_string())?;
    let mesh = gltf
        .meshes()
        .next()
        .ok_or_else(|| "GLB has no meshes".to_string())?;
    let primitive = mesh
        .primitives()
        .next()
        .ok_or_else(|| "GLB mesh has no primitives".to_string())?;
    let reader = primitive.reader(|_buffer| Some(blob));
    let vertices = reader
        .read_positions()
        .ok_or_else(|| "GLB missing POSITION data".to_string())?
        .collect::<Vec<_>>();
    if vertices.is_empty() {
        return Err("GLB mesh has no vertices".to_string());
    }
    let indices = reader
        .read_indices()
        .ok_or_else(|| "GLB missing index data".to_string())?
        .into_u32()
        .collect::<Vec<_>>();
    if !indices.len().is_multiple_of(3) {
        return Err("GLB indices are not triangles".to_string());
    }
    let vertex_count = vertices.len() as u32;
    let mut faces = Vec::with_capacity(indices.len() / 3);
    for tri in indices.as_chunks::<3>().0 {
        if tri.iter().any(|index| *index >= vertex_count) {
            return Err("GLB indices reference out-of-range vertices".to_string());
        }
        faces.push(*tri);
    }
    let mut normals = reader
        .read_normals()
        .map(|normals| normals.collect::<Vec<_>>())
        .unwrap_or_default();
    if normals.len() != vertices.len() {
        normals.clear();
    }
    Ok(RenderMesh {
        vertices,
        faces,
        normals,
    })
}

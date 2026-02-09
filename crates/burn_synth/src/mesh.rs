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

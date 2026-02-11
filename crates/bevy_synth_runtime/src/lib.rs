#![recursion_limit = "256"]

pub mod args;
pub mod cache;
pub mod io;
pub mod mesh;
pub mod model_loader;
pub mod paths;
pub mod state;
pub mod worker;

pub use burn_tripo::pipeline::mesh::Mesh as TripoMesh;

#[derive(Clone, Copy, Debug)]
pub struct SynthMeshMaterial {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha: f32,
}

#[derive(Clone, Debug)]
pub struct SynthMeshTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct SynthMeshPbrTextures {
    pub base_color: SynthMeshTexture,
    pub metallic_roughness: SynthMeshTexture,
    pub normal: Option<SynthMeshTexture>,
    pub emissive: Option<SynthMeshTexture>,
    pub occlusion: Option<SynthMeshTexture>,
}

#[derive(Clone, Debug)]
pub struct SynthMesh {
    pub mesh: TripoMesh,
    pub uvs: Vec<[f32; 2]>,
    pub material: Option<SynthMeshMaterial>,
    pub pbr_textures: Option<SynthMeshPbrTextures>,
}

impl From<TripoMesh> for SynthMesh {
    fn from(mesh: TripoMesh) -> Self {
        Self {
            mesh,
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        }
    }
}

impl From<burn_trellis::Mesh> for SynthMesh {
    fn from(mesh: burn_trellis::Mesh) -> Self {
        Self {
            mesh: TripoMesh {
                vertices: mesh.vertices,
                faces: mesh.faces,
            },
            uvs: mesh.uvs,
            material: mesh.material.map(|material| SynthMeshMaterial {
                base_color: material.base_color,
                metallic: material.metallic,
                roughness: material.roughness,
                alpha: material.alpha,
            }),
            pbr_textures: mesh.pbr_textures.map(|textures| SynthMeshPbrTextures {
                base_color: SynthMeshTexture {
                    width: textures.base_color.width,
                    height: textures.base_color.height,
                    rgba8: textures.base_color.rgba8,
                },
                metallic_roughness: SynthMeshTexture {
                    width: textures.metallic_roughness.width,
                    height: textures.metallic_roughness.height,
                    rgba8: textures.metallic_roughness.rgba8,
                },
                normal: textures.normal.map(|texture| SynthMeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
                emissive: textures.emissive.map(|texture| SynthMeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
                occlusion: textures.occlusion.map(|texture| SynthMeshTexture {
                    width: texture.width,
                    height: texture.height,
                    rgba8: texture.rgba8,
                }),
            }),
        }
    }
}

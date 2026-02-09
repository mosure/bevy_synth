use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshMaterial {
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    pub alpha: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshTexture {
    pub width: u32,
    pub height: u32,
    pub rgba8: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshPbrTextures {
    pub base_color: MeshTexture,
    pub metallic_roughness: MeshTexture,
    pub normal: Option<MeshTexture>,
    pub emissive: Option<MeshTexture>,
    pub occlusion: Option<MeshTexture>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub vertices: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub material: Option<MeshMaterial>,
    pub pbr_textures: Option<MeshPbrTextures>,
}

impl Mesh {
    pub fn new(vertices: Vec<[f32; 3]>, faces: Vec<[u32; 3]>) -> Self {
        Self {
            vertices,
            faces,
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        }
    }

    pub fn with_material(
        vertices: Vec<[f32; 3]>,
        faces: Vec<[u32; 3]>,
        material: MeshMaterial,
    ) -> Self {
        Self {
            vertices,
            faces,
            uvs: Vec::new(),
            material: Some(material),
            pbr_textures: None,
        }
    }

    pub fn with_pbr(mut self, uvs: Vec<[f32; 2]>, textures: MeshPbrTextures) -> Self {
        self.uvs = uvs;
        self.pbr_textures = Some(textures);
        self
    }

    pub fn has_pbr_textures(&self) -> bool {
        self.pbr_textures.is_some() && self.uvs.len() == self.vertices.len()
    }

    pub fn clear_pbr(&mut self) {
        self.pbr_textures = None;
        self.uvs.clear();
    }

    pub fn ensure_uvs(&mut self) {
        if self.uvs.len() == self.vertices.len() {
            return;
        }
        self.uvs = vec![[0.0, 0.0]; self.vertices.len()];
    }

    pub fn with_default_uvs(mut self) -> Self {
        self.ensure_uvs();
        self
    }

    pub fn pbr_resolution(&self) -> Option<(u32, u32)> {
        self.pbr_textures
            .as_ref()
            .map(|textures| (textures.base_color.width, textures.base_color.height))
    }

    pub fn fallback_material(&self) -> Option<MeshMaterial> {
        self.material
    }

    pub fn replace_material(&mut self, material: Option<MeshMaterial>) {
        self.material = material;
    }

    pub fn map_uvs(mut self, mut f: impl FnMut([f32; 2]) -> [f32; 2]) -> Self {
        for uv in &mut self.uvs {
            *uv = f(*uv);
        }
        self
    }

    pub fn texture_texel_count(&self) -> usize {
        self.pbr_textures
            .as_ref()
            .map(|textures| {
                (textures.base_color.width as usize) * (textures.base_color.height as usize)
            })
            .unwrap_or(0)
    }
}

pub fn load_obj_mesh(path: &Path) -> Result<Mesh, String> {
    let file =
        fs::File::open(path).map_err(|err| format!("failed to open {}: {err}", path.display()))?;
    let reader = BufReader::new(file);
    let mut vertices = Vec::new();
    let mut faces = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|err| format!("failed to read OBJ line: {err}"))?;
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("v ") {
            let parts = rest.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 3 {
                continue;
            }
            let x = parts[0]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex x '{}': {err}", parts[0]))?;
            let y = parts[1]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex y '{}': {err}", parts[1]))?;
            let z = parts[2]
                .parse::<f32>()
                .map_err(|err| format!("invalid OBJ vertex z '{}': {err}", parts[2]))?;
            vertices.push([x, y, z]);
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("f ") {
            let parts = rest.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 3 {
                continue;
            }
            let mut idx = [0u32; 3];
            for i in 0..3 {
                let value = parts[i]
                    .split('/')
                    .next()
                    .ok_or_else(|| format!("invalid OBJ face index '{}'", parts[i]))?;
                let parsed = value
                    .parse::<u32>()
                    .map_err(|err| format!("invalid OBJ face index '{}': {err}", value))?;
                idx[i] = parsed.saturating_sub(1);
            }
            faces.push(idx);
        }
    }

    if vertices.is_empty() || faces.is_empty() {
        return Err(format!(
            "OBJ '{}' did not contain vertices/faces",
            path.display()
        ));
    }
    Ok(Mesh {
        vertices,
        faces,
        uvs: Vec::new(),
        material: None,
        pbr_textures: None,
    })
}

pub fn write_obj_mesh(path: &Path, mesh: &Mesh) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    let file = fs::File::create(path)
        .map_err(|err| format!("failed to create '{}': {err}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for vertex in &mesh.vertices {
        writeln!(writer, "v {} {} {}", vertex[0], vertex[1], vertex[2])
            .map_err(|err| format!("failed to write vertex: {err}"))?;
    }
    for face in &mesh.faces {
        writeln!(writer, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)
            .map_err(|err| format!("failed to write face: {err}"))?;
    }
    writer
        .flush()
        .map_err(|err| format!("failed to flush OBJ: {err}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{Mesh, load_obj_mesh, write_obj_mesh};

    #[test]
    fn obj_roundtrip_works() {
        let mesh = Mesh {
            vertices: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            faces: vec![[0, 1, 2]],
            uvs: Vec::new(),
            material: None,
            pbr_textures: None,
        };
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("burn_trellis_mesh_{unique}.obj"));
        write_obj_mesh(&path, &mesh).expect("failed to write obj");
        let loaded = load_obj_mesh(&path).expect("failed to read obj");
        assert_eq!(loaded, mesh);
        let _ = std::fs::remove_file(path);
    }
}

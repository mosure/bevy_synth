use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::json;

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

pub fn write_glb_mesh(path: &Path, mesh: &Mesh) -> Result<(), String> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err("cannot export empty mesh".to_string());
    }
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }

    let mut positions = Vec::with_capacity(mesh.vertices.len() * 12);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
        positions.extend_from_slice(&vertex[0].to_le_bytes());
        positions.extend_from_slice(&vertex[1].to_le_bytes());
        positions.extend_from_slice(&vertex[2].to_le_bytes());
    }

    let mut indices = Vec::with_capacity(mesh.faces.len() * 12);
    for face in &mesh.faces {
        indices.extend_from_slice(&face[0].to_le_bytes());
        indices.extend_from_slice(&face[1].to_le_bytes());
        indices.extend_from_slice(&face[2].to_le_bytes());
    }

    let mut bin = Vec::with_capacity(positions.len() + indices.len() + 8);
    let position_offset = 0usize;
    bin.extend_from_slice(positions.as_slice());
    pad_buffer_to_4_bytes(&mut bin, 0);
    let position_len = positions.len();
    let index_offset = bin.len();
    bin.extend_from_slice(indices.as_slice());
    let index_len = indices.len();
    pad_buffer_to_4_bytes(&mut bin, 0);

    let mut primitive = serde_json::Map::new();
    primitive.insert("attributes".to_string(), json!({ "POSITION": 0 }));
    primitive.insert("indices".to_string(), json!(1));
    primitive.insert("mode".to_string(), json!(4));

    if let Some(material) = mesh.material {
        primitive.insert("material".to_string(), json!(0));
        let json_value = json!({
            "asset": { "version": "2.0", "generator": "burn_trellis" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [primitive] }],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [material.base_color[0], material.base_color[1], material.base_color[2], material.alpha],
                    "metallicFactor": material.metallic,
                    "roughnessFactor": material.roughness
                }
            }],
            "buffers": [{ "byteLength": bin.len() }],
            "bufferViews": [
                { "buffer": 0, "byteOffset": position_offset, "byteLength": position_len, "target": 34962 },
                { "buffer": 0, "byteOffset": index_offset, "byteLength": index_len, "target": 34963 }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,
                    "count": mesh.vertices.len(),
                    "type": "VEC3",
                    "min": [min[0], min[1], min[2]],
                    "max": [max[0], max[1], max[2]]
                },
                {
                    "bufferView": 1,
                    "componentType": 5125,
                    "count": mesh.faces.len() * 3,
                    "type": "SCALAR"
                }
            ]
        });
        return write_glb_bytes(path, json_value, bin);
    }

    let json_value = json!({
        "asset": { "version": "2.0", "generator": "burn_trellis" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [primitive] }],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": position_offset, "byteLength": position_len, "target": 34962 },
            { "buffer": 0, "byteOffset": index_offset, "byteLength": index_len, "target": 34963 }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": mesh.vertices.len(),
                "type": "VEC3",
                "min": [min[0], min[1], min[2]],
                "max": [max[0], max[1], max[2]]
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": mesh.faces.len() * 3,
                "type": "SCALAR"
            }
        ]
    });
    write_glb_bytes(path, json_value, bin)
}

fn write_glb_bytes(
    path: &Path,
    json_value: serde_json::Value,
    mut bin: Vec<u8>,
) -> Result<(), String> {
    let mut json_bytes = serde_json::to_vec(&json_value)
        .map_err(|err| format!("failed to serialize glb json: {err}"))?;
    pad_buffer_to_4_bytes(&mut json_bytes, 0x20);
    pad_buffer_to_4_bytes(&mut bin, 0);

    let json_chunk_len = json_bytes.len() as u32;
    let bin_chunk_len = bin.len() as u32;
    let total_len = 12u32 + 8 + json_chunk_len + 8 + bin_chunk_len;

    let mut glb = Vec::with_capacity(total_len as usize);
    glb.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&total_len.to_le_bytes());
    glb.extend_from_slice(&json_chunk_len.to_le_bytes());
    glb.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
    glb.extend_from_slice(json_bytes.as_slice());
    glb.extend_from_slice(&bin_chunk_len.to_le_bytes());
    glb.extend_from_slice(&0x004E_4942u32.to_le_bytes());
    glb.extend_from_slice(bin.as_slice());

    fs::write(path, glb).map_err(|err| format!("failed to write '{}': {err}", path.display()))
}

fn pad_buffer_to_4_bytes(buffer: &mut Vec<u8>, byte: u8) {
    let remainder = buffer.len() % 4;
    if remainder == 0 {
        return;
    }
    let pad = 4 - remainder;
    buffer.extend(std::iter::repeat_n(byte, pad));
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::{Mesh, load_obj_mesh, write_glb_mesh, write_obj_mesh};

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

    #[test]
    fn glb_writer_emits_glb_header() {
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
        let path = std::env::temp_dir().join(format!("burn_trellis_mesh_{unique}.glb"));
        write_glb_mesh(&path, &mesh).expect("failed to write glb");
        let bytes = std::fs::read(&path).expect("failed to read glb");
        assert!(bytes.len() >= 12, "glb must contain header");
        assert_eq!(&bytes[0..4], b"glTF");
        let _ = std::fs::remove_file(path);
    }
}

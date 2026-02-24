use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use image::ImageEncoder;
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

#[derive(Clone, Debug)]
struct MeshBinaryLayout {
    buffer: Vec<u8>,
    positions_byte_offset: usize,
    positions_byte_length: usize,
    indices_byte_offset: usize,
    indices_byte_length: usize,
    uvs_byte_offset: Option<usize>,
    uvs_byte_length: Option<usize>,
    base_color_image_view: Option<(usize, usize)>,
    metallic_roughness_image_view: Option<(usize, usize)>,
    normal_image_view: Option<(usize, usize)>,
    emissive_image_view: Option<(usize, usize)>,
    occlusion_image_view: Option<(usize, usize)>,
    min: [f32; 3],
    max: [f32; 3],
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
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    let layout = build_mesh_binary_layout(mesh)?;
    let json_value = gltf_json(mesh, &layout);
    write_glb_bytes(path, json_value, layout.buffer)
}

fn build_mesh_binary_layout(mesh: &Mesh) -> Result<MeshBinaryLayout, String> {
    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err("cannot export empty mesh".to_string());
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }

    let mut buffer = Vec::with_capacity(mesh.vertices.len() * 12 + mesh.faces.len() * 12 + 8192);
    let positions_byte_offset = buffer.len();
    for vertex in &mesh.vertices {
        for component in vertex {
            buffer.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_byte_length = buffer.len() - positions_byte_offset;

    let mut uvs_byte_offset = None;
    let mut uvs_byte_length = None;
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        pad_buffer_to_4_bytes(&mut buffer, 0);
        let offset = buffer.len();
        for uv in &mesh.uvs {
            buffer.extend_from_slice(&uv[0].to_le_bytes());
            buffer.extend_from_slice(&uv[1].to_le_bytes());
        }
        uvs_byte_offset = Some(offset);
        uvs_byte_length = Some(buffer.len() - offset);
    }

    pad_buffer_to_4_bytes(&mut buffer, 0);
    let indices_byte_offset = buffer.len();
    for face in &mesh.faces {
        for index in face {
            buffer.extend_from_slice(&index.to_le_bytes());
        }
    }
    let indices_byte_length = buffer.len() - indices_byte_offset;

    let mut base_color_image_view = None;
    let mut metallic_roughness_image_view = None;
    let mut normal_image_view = None;
    let mut emissive_image_view = None;
    let mut occlusion_image_view = None;
    if let Some(pbr) = mesh.pbr_textures.as_ref() {
        let base_png = encode_rgba_texture_png(&pbr.base_color)?;
        let mr_png = encode_rgba_texture_png(&pbr.metallic_roughness)?;

        pad_buffer_to_4_bytes(&mut buffer, 0);
        let base_offset = buffer.len();
        buffer.extend_from_slice(base_png.as_slice());
        base_color_image_view = Some((base_offset, base_png.len()));

        pad_buffer_to_4_bytes(&mut buffer, 0);
        let mr_offset = buffer.len();
        buffer.extend_from_slice(mr_png.as_slice());
        metallic_roughness_image_view = Some((mr_offset, mr_png.len()));

        if let Some(normal) = pbr.normal.as_ref() {
            let png = encode_rgba_texture_png(normal)?;
            pad_buffer_to_4_bytes(&mut buffer, 0);
            let offset = buffer.len();
            buffer.extend_from_slice(png.as_slice());
            normal_image_view = Some((offset, png.len()));
        }
        if let Some(emissive) = pbr.emissive.as_ref() {
            let png = encode_rgba_texture_png(emissive)?;
            pad_buffer_to_4_bytes(&mut buffer, 0);
            let offset = buffer.len();
            buffer.extend_from_slice(png.as_slice());
            emissive_image_view = Some((offset, png.len()));
        }
        if let Some(occlusion) = pbr.occlusion.as_ref() {
            let png = encode_rgba_texture_png(occlusion)?;
            pad_buffer_to_4_bytes(&mut buffer, 0);
            let offset = buffer.len();
            buffer.extend_from_slice(png.as_slice());
            occlusion_image_view = Some((offset, png.len()));
        }
    }

    Ok(MeshBinaryLayout {
        buffer,
        positions_byte_offset,
        positions_byte_length,
        indices_byte_offset,
        indices_byte_length,
        uvs_byte_offset,
        uvs_byte_length,
        base_color_image_view,
        metallic_roughness_image_view,
        normal_image_view,
        emissive_image_view,
        occlusion_image_view,
        min,
        max,
    })
}

fn gltf_json(mesh: &Mesh, layout: &MeshBinaryLayout) -> serde_json::Value {
    let mut primitive = serde_json::Map::new();
    primitive.insert("attributes".to_string(), json!({ "POSITION": 0 }));
    primitive.insert("indices".to_string(), json!(1));
    primitive.insert("mode".to_string(), json!(4));
    if mesh.uvs.len() == mesh.vertices.len()
        && !mesh.uvs.is_empty()
        && let Some(attributes) = primitive
            .get_mut("attributes")
            .and_then(|value| value.as_object_mut())
    {
        attributes.insert("TEXCOORD_0".to_string(), json!(2));
    }

    let mut buffer_views = Vec::new();
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": layout.positions_byte_offset,
        "byteLength": layout.positions_byte_length,
        "target": 34962
    }));
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": layout.indices_byte_offset,
        "byteLength": layout.indices_byte_length,
        "target": 34963
    }));
    if let (Some(uv_offset), Some(uv_len)) = (layout.uvs_byte_offset, layout.uvs_byte_length) {
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": uv_len,
            "target": 34962
        }));
    }

    let mut accessors = Vec::new();
    accessors.push(json!({
        "bufferView": 0,
        "componentType": 5126,
        "count": mesh.vertices.len(),
        "type": "VEC3",
        "min": [layout.min[0], layout.min[1], layout.min[2]],
        "max": [layout.max[0], layout.max[1], layout.max[2]]
    }));
    accessors.push(json!({
        "bufferView": 1,
        "componentType": 5125,
        "count": mesh.faces.len() * 3,
        "type": "SCALAR"
    }));
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        accessors.push(json!({
            "bufferView": 2,
            "componentType": 5126,
            "count": mesh.uvs.len(),
            "type": "VEC2"
        }));
    }

    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut materials = Vec::new();
    let mut pbr_mr = json!({});
    let mut push_texture_image = |byte_offset: usize, byte_length: usize| -> usize {
        let view_index = buffer_views.len();
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length
        }));
        let image_index = images.len();
        images.push(json!({
            "bufferView": view_index,
            "mimeType": "image/png"
        }));
        let texture_index = textures.len();
        textures.push(json!({ "source": image_index }));
        texture_index
    };

    if let Some(material) = mesh.material {
        pbr_mr = json!({
            "baseColorFactor": [
                material.base_color[0],
                material.base_color[1],
                material.base_color[2],
                material.alpha.clamp(0.0, 1.0)
            ],
            "metallicFactor": material.metallic.clamp(0.0, 1.0),
            "roughnessFactor": material.roughness.clamp(0.0, 1.0)
        });
    }
    if let Some((base_offset, base_len)) = layout.base_color_image_view {
        let texture_index = push_texture_image(base_offset, base_len);
        pbr_mr["baseColorTexture"] = json!({ "index": texture_index });
    }
    if let Some((mr_offset, mr_len)) = layout.metallic_roughness_image_view {
        let texture_index = push_texture_image(mr_offset, mr_len);
        pbr_mr["metallicRoughnessTexture"] = json!({ "index": texture_index });
    }

    if mesh.material.is_some() || mesh.pbr_textures.is_some() {
        let alpha = mesh
            .material
            .map(|value| value.alpha)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let mut material = json!({
            "pbrMetallicRoughness": pbr_mr,
            "alphaMode": if alpha < 0.995 { "BLEND" } else { "OPAQUE" },
            "doubleSided": true
        });
        if let Some((normal_offset, normal_len)) = layout.normal_image_view {
            let texture_index = push_texture_image(normal_offset, normal_len);
            material["normalTexture"] = json!({ "index": texture_index });
        }
        if let Some((emissive_offset, emissive_len)) = layout.emissive_image_view {
            let texture_index = push_texture_image(emissive_offset, emissive_len);
            material["emissiveTexture"] = json!({ "index": texture_index });
            material["emissiveFactor"] = json!([1.0, 1.0, 1.0]);
        }
        if let Some((occlusion_offset, occlusion_len)) = layout.occlusion_image_view {
            let texture_index = push_texture_image(occlusion_offset, occlusion_len);
            material["occlusionTexture"] = json!({ "index": texture_index });
        }
        let material_index = materials.len();
        materials.push(material);
        primitive.insert("material".to_string(), json!(material_index));
    }

    let mut gltf = json!({
        "asset": { "version": "2.0", "generator": "burn_trellis" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{ "primitives": [primitive] }],
        "buffers": [{ "byteLength": layout.buffer.len() }],
        "bufferViews": buffer_views,
        "accessors": accessors
    });
    if !materials.is_empty() {
        gltf["materials"] = serde_json::Value::Array(materials);
    }
    if !images.is_empty() {
        gltf["images"] = serde_json::Value::Array(images);
    }
    if !textures.is_empty() {
        gltf["textures"] = serde_json::Value::Array(textures);
    }
    gltf
}

fn encode_rgba_texture_png(texture: &MeshTexture) -> Result<Vec<u8>, String> {
    let expected = texture.width as usize * texture.height as usize * 4;
    if texture.rgba8.len() != expected {
        return Err(format!(
            "texture byte length mismatch: expected {}, got {}",
            expected,
            texture.rgba8.len()
        ));
    }

    let mut out = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut out);
    encoder
        .write_image(
            texture.rgba8.as_slice(),
            texture.width,
            texture.height,
            image::ColorType::Rgba8.into(),
        )
        .map_err(|err| format!("failed to encode texture png: {err}"))?;
    Ok(out)
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

    use super::{
        Mesh, MeshPbrTextures, MeshTexture, load_obj_mesh, write_glb_mesh, write_obj_mesh,
    };

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

    #[test]
    fn glb_writer_embeds_pbr_textures_when_present() {
        let mesh = Mesh {
            vertices: vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]],
            faces: vec![[0, 1, 2]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            material: None,
            pbr_textures: Some(MeshPbrTextures {
                base_color: MeshTexture {
                    width: 1,
                    height: 1,
                    rgba8: vec![255, 0, 0, 255],
                },
                metallic_roughness: MeshTexture {
                    width: 1,
                    height: 1,
                    rgba8: vec![128, 128, 0, 255],
                },
                normal: None,
                emissive: None,
                occlusion: None,
            }),
        };
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock drift")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("burn_trellis_mesh_{unique}_pbr.glb"));
        write_glb_mesh(&path, &mesh).expect("failed to write glb");
        let bytes = std::fs::read(&path).expect("failed to read glb");
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().expect("json len")) as usize;
        let json_start = 20usize;
        let json_end = json_start + json_len;
        let json_bytes = &bytes[json_start..json_end];
        let json_text = std::str::from_utf8(json_bytes)
            .expect("json utf8")
            .trim_end_matches(' ');
        let value: serde_json::Value = serde_json::from_str(json_text).expect("json parse");
        assert!(
            value.get("images").is_some(),
            "expected embedded image views"
        );
        assert!(
            value.get("textures").is_some(),
            "expected embedded texture table"
        );
        assert!(
            value.get("materials").is_some(),
            "expected textured material data"
        );
        let primitive = &value["meshes"][0]["primitives"][0];
        assert_eq!(primitive["attributes"]["TEXCOORD_0"], 2);
        let _ = std::fs::remove_file(path);
    }
}

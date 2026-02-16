use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub enum ImageSource {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

impl ImageSource {
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self::Path(path.into())
    }

    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self::Bytes(bytes.into())
    }

    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => Some(path.as_path()),
            Self::Bytes(_) => None,
        }
    }

    pub fn bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Path(_) => None,
            Self::Bytes(bytes) => Some(bytes),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TextPrompt(pub String);

impl From<String> for TextPrompt {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for TextPrompt {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl TextPrompt {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use std::borrow::Cow;
#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use std::fs;
#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use std::io;

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use crate::mesh::{Mesh, MeshTexture};
#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use image::ImageEncoder;
#[cfg(any(feature = "runtime", feature = "wasm-api"))]
use serde_json::{Value, json};

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
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

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
pub fn write_glb_mesh(path: &Path, mesh: &Mesh) -> Result<(), String> {
    ensure_parent_dir(path).map_err(|err| err.to_string())?;
    let bytes = mesh_to_glb_bytes(mesh)?;
    fs::write(path, bytes).map_err(|err| format!("failed to write {}: {err}", path.display()))
}

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
pub fn mesh_to_glb_bytes(mesh: &Mesh) -> Result<Vec<u8>, String> {
    let layout = build_mesh_binary_layout(mesh)?;
    let gltf = gltf_json(mesh, &layout);
    let json_bytes = serde_json::to_vec(&gltf)
        .map_err(|err| format!("failed to serialize glTF json chunk: {err}"))?;
    gltf::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(json_bytes),
        bin: Some(Cow::Owned(layout.buffer)),
    }
    .to_vec()
    .map_err(|err| format!("failed to build GLB: {err}"))
}

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
fn build_mesh_binary_layout(mesh: &Mesh) -> Result<MeshBinaryLayout, String> {
    if mesh.vertices.is_empty() {
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
        pad_buffer_4(&mut buffer);
        let offset = buffer.len();
        for uv in &mesh.uvs {
            buffer.extend_from_slice(&uv[0].to_le_bytes());
            buffer.extend_from_slice(&uv[1].to_le_bytes());
        }
        uvs_byte_offset = Some(offset);
        uvs_byte_length = Some(buffer.len() - offset);
    }

    pad_buffer_4(&mut buffer);
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
        pad_buffer_4(&mut buffer);
        let base_offset = buffer.len();
        buffer.extend_from_slice(base_png.as_slice());
        base_color_image_view = Some((base_offset, base_png.len()));
        pad_buffer_4(&mut buffer);
        let mr_offset = buffer.len();
        buffer.extend_from_slice(mr_png.as_slice());
        metallic_roughness_image_view = Some((mr_offset, mr_png.len()));

        if let Some(normal) = pbr.normal.as_ref() {
            let png = encode_rgba_texture_png(normal)?;
            pad_buffer_4(&mut buffer);
            let offset = buffer.len();
            buffer.extend_from_slice(png.as_slice());
            normal_image_view = Some((offset, png.len()));
        }
        if let Some(emissive) = pbr.emissive.as_ref() {
            let png = encode_rgba_texture_png(emissive)?;
            pad_buffer_4(&mut buffer);
            let offset = buffer.len();
            buffer.extend_from_slice(png.as_slice());
            emissive_image_view = Some((offset, png.len()));
        }
        if let Some(occlusion) = pbr.occlusion.as_ref() {
            let png = encode_rgba_texture_png(occlusion)?;
            pad_buffer_4(&mut buffer);
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

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
fn gltf_json(mesh: &Mesh, layout: &MeshBinaryLayout) -> Value {
    let mut primitive = json!({
        "attributes": {
            "POSITION": 0
        },
        "indices": 1,
        "mode": 4
    });
    if mesh.uvs.len() == mesh.vertices.len() && !mesh.uvs.is_empty() {
        primitive["attributes"]["TEXCOORD_0"] = json!(2);
    }

    let buffers = vec![json!({
        "byteLength": layout.buffer.len(),
    })];

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
        "min": layout.min,
        "max": layout.max
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
        let material_index = materials.len();
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
        materials.push(material);
        primitive["material"] = json!(material_index);
    }

    let mut gltf = json!({
        "asset": {
            "version": "2.0",
            "generator": "burn_synth"
        },
        "scene": 0,
        "scenes": [
            { "nodes": [0] }
        ],
        "nodes": [
            { "mesh": 0 }
        ],
        "meshes": [
            {
                "primitives": [
                    primitive
                ]
            }
        ],
        "buffers": buffers,
        "bufferViews": buffer_views,
        "accessors": accessors
    });
    if !materials.is_empty() {
        gltf["materials"] = Value::Array(materials);
    }
    if !images.is_empty() {
        gltf["images"] = Value::Array(images);
    }
    if !textures.is_empty() {
        gltf["textures"] = Value::Array(textures);
    }
    gltf
}

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
fn pad_buffer_4(buffer: &mut Vec<u8>) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
}

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
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

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
fn ensure_parent_dir(path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(all(test, feature = "runtime"))]
mod tests {
    use super::*;
    use crate::mesh::{Mesh, MeshMaterial, MeshPbrTextures, MeshTexture};

    fn test_texture(width: u32, height: u32, rgba: [u8; 4]) -> MeshTexture {
        let mut bytes = Vec::with_capacity(width as usize * height as usize * 4);
        for _ in 0..(width as usize * height as usize) {
            bytes.extend_from_slice(&rgba);
        }
        MeshTexture {
            width,
            height,
            rgba8: bytes,
        }
    }

    fn sample_mesh_with_pbr() -> Mesh {
        Mesh {
            vertices: vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.8, 0.0]],
            faces: vec![[0, 1, 2]],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            material: Some(MeshMaterial {
                base_color: [1.0, 1.0, 1.0],
                metallic: 1.0,
                roughness: 1.0,
                alpha: 1.0,
            }),
            pbr_textures: Some(MeshPbrTextures {
                base_color: test_texture(2, 2, [200, 180, 160, 255]),
                metallic_roughness: test_texture(2, 2, [0, 128, 64, 255]),
                normal: None,
                emissive: None,
                occlusion: None,
            }),
        }
    }

    #[test]
    fn glb_embeds_pbr_textures_when_present() {
        let mesh = sample_mesh_with_pbr();
        let bytes = mesh_to_glb_bytes(&mesh).expect("glb export");
        let glb = gltf::Glb::from_slice(bytes.as_slice()).expect("parse glb");
        let json: Value = serde_json::from_slice(glb.json.as_ref()).expect("parse glb json");

        let materials = json["materials"].as_array().expect("materials array");
        assert_eq!(materials.len(), 1);
        let pbr = &materials[0]["pbrMetallicRoughness"];
        assert!(pbr.get("baseColorTexture").is_some());
        assert!(pbr.get("metallicRoughnessTexture").is_some());
        assert!(
            json["textures"]
                .as_array()
                .is_some_and(|value| !value.is_empty())
        );
        assert!(
            json["images"]
                .as_array()
                .is_some_and(|value| !value.is_empty())
        );
    }

    #[test]
    fn glb_writes_material_when_only_textures_are_present() {
        let mut mesh = sample_mesh_with_pbr();
        mesh.material = None;

        let bytes = mesh_to_glb_bytes(&mesh).expect("glb export");
        let glb = gltf::Glb::from_slice(bytes.as_slice()).expect("parse glb");
        let json: Value = serde_json::from_slice(glb.json.as_ref()).expect("parse glb json");

        let materials = json["materials"].as_array().expect("materials array");
        assert_eq!(materials.len(), 1);
        assert_eq!(materials[0]["alphaMode"], "OPAQUE");
        let pbr = &materials[0]["pbrMetallicRoughness"];
        assert!(pbr.get("baseColorTexture").is_some());
        assert!(pbr.get("metallicRoughnessTexture").is_some());
    }
}

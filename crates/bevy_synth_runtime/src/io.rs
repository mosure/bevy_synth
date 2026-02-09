use std::borrow::Cow;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use image::ImageEncoder;
use safetensors::tensor::{SafeTensors, TensorView};

use crate::{SynthMesh, SynthMeshTexture};

pub fn write_glb(path: &Path, mesh: &SynthMesh) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let glb = mesh_to_glb_bytes(mesh)?;
    fs::write(path, glb)?;
    Ok(())
}

pub fn mesh_to_glb_bytes(mesh: &SynthMesh) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let layout = build_mesh_binary_layout(mesh)?;
    let gltf = gltf_json(mesh, &layout);
    let json_bytes = serde_json::to_vec(&gltf)?;
    let glb = gltf::Glb {
        header: gltf::binary::Header {
            magic: *b"glTF",
            version: 2,
            length: 0,
        },
        json: Cow::Owned(json_bytes),
        bin: Some(Cow::Owned(layout.buffer)),
    }
    .to_vec()?;
    Ok(glb)
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

fn build_mesh_binary_layout(
    mesh: &SynthMesh,
) -> Result<MeshBinaryLayout, Box<dyn std::error::Error>> {
    if mesh.mesh.vertices.is_empty() {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            "cannot export empty mesh",
        )));
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in &mesh.mesh.vertices {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }

    let mut buffer =
        Vec::with_capacity(mesh.mesh.vertices.len() * 12 + mesh.mesh.faces.len() * 12 + 8192);
    let positions_byte_offset = buffer.len();
    for vertex in &mesh.mesh.vertices {
        for component in vertex {
            buffer.extend_from_slice(&component.to_le_bytes());
        }
    }
    let positions_byte_length = buffer.len();

    let mut uvs_byte_offset = None;
    let mut uvs_byte_length = None;
    if mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty() {
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
    for face in &mesh.mesh.faces {
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
            let normal_png = encode_rgba_texture_png(normal)?;
            pad_buffer_4(&mut buffer);
            let normal_offset = buffer.len();
            buffer.extend_from_slice(normal_png.as_slice());
            normal_image_view = Some((normal_offset, normal_png.len()));
        }
        if let Some(emissive) = pbr.emissive.as_ref() {
            let emissive_png = encode_rgba_texture_png(emissive)?;
            pad_buffer_4(&mut buffer);
            let emissive_offset = buffer.len();
            buffer.extend_from_slice(emissive_png.as_slice());
            emissive_image_view = Some((emissive_offset, emissive_png.len()));
        }
        if let Some(occlusion) = pbr.occlusion.as_ref() {
            let occlusion_png = encode_rgba_texture_png(occlusion)?;
            pad_buffer_4(&mut buffer);
            let occlusion_offset = buffer.len();
            buffer.extend_from_slice(occlusion_png.as_slice());
            occlusion_image_view = Some((occlusion_offset, occlusion_png.len()));
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

fn gltf_json(mesh: &SynthMesh, layout: &MeshBinaryLayout) -> serde_json::Value {
    let mut primitive = serde_json::json!({
        "attributes": {
            "POSITION": 0
        },
        "indices": 1,
        "mode": 4
    });
    if mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty() {
        primitive["attributes"]["TEXCOORD_0"] = serde_json::json!(2);
    }

    let buffers = vec![serde_json::json!({
        "byteLength": layout.buffer.len(),
    })];

    let mut buffer_views = Vec::new();
    buffer_views.push(serde_json::json!({
        "buffer": 0,
        "byteOffset": layout.positions_byte_offset,
        "byteLength": layout.positions_byte_length,
        "target": 34962
    }));
    buffer_views.push(serde_json::json!({
        "buffer": 0,
        "byteOffset": layout.indices_byte_offset,
        "byteLength": layout.indices_byte_length,
        "target": 34963
    }));
    if let (Some(uv_offset), Some(uv_len)) = (layout.uvs_byte_offset, layout.uvs_byte_length) {
        buffer_views.push(serde_json::json!({
            "buffer": 0,
            "byteOffset": uv_offset,
            "byteLength": uv_len,
            "target": 34962
        }));
    }

    let mut accessors = Vec::new();
    accessors.push(serde_json::json!({
        "bufferView": 0,
        "componentType": 5126,
        "count": mesh.mesh.vertices.len(),
        "type": "VEC3",
        "min": layout.min,
        "max": layout.max
    }));
    accessors.push(serde_json::json!({
        "bufferView": 1,
        "componentType": 5125,
        "count": mesh.mesh.faces.len() * 3,
        "type": "SCALAR"
    }));
    if mesh.uvs.len() == mesh.mesh.vertices.len() && !mesh.uvs.is_empty() {
        accessors.push(serde_json::json!({
            "bufferView": 2,
            "componentType": 5126,
            "count": mesh.uvs.len(),
            "type": "VEC2"
        }));
    }

    let mut images = Vec::new();
    let mut textures = Vec::new();
    let mut materials = Vec::new();
    let mut pbr_mr = serde_json::json!({});
    let mut push_texture_image = |byte_offset: usize, byte_length: usize| -> usize {
        let view_index = buffer_views.len();
        buffer_views.push(serde_json::json!({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length
        }));
        let image_index = images.len();
        images.push(serde_json::json!({
            "bufferView": view_index,
            "mimeType": "image/png"
        }));
        let texture_index = textures.len();
        textures.push(serde_json::json!({ "source": image_index }));
        texture_index
    };
    if let Some(material) = mesh.material {
        pbr_mr = serde_json::json!({
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
        pbr_mr["baseColorTexture"] = serde_json::json!({ "index": texture_index });
    }
    if let Some((mr_offset, mr_len)) = layout.metallic_roughness_image_view {
        let texture_index = push_texture_image(mr_offset, mr_len);
        pbr_mr["metallicRoughnessTexture"] = serde_json::json!({ "index": texture_index });
    }

    if mesh.material.is_some() || mesh.pbr_textures.is_some() {
        let alpha = mesh
            .material
            .map(|value| value.alpha)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let material_index = materials.len();
        let mut material = serde_json::json!({
            "pbrMetallicRoughness": pbr_mr,
            "alphaMode": if alpha < 0.995 { "BLEND" } else { "OPAQUE" },
            "doubleSided": true
        });
        if let Some((normal_offset, normal_len)) = layout.normal_image_view {
            let texture_index = push_texture_image(normal_offset, normal_len);
            material["normalTexture"] = serde_json::json!({ "index": texture_index });
        }
        if let Some((emissive_offset, emissive_len)) = layout.emissive_image_view {
            let texture_index = push_texture_image(emissive_offset, emissive_len);
            material["emissiveTexture"] = serde_json::json!({ "index": texture_index });
            material["emissiveFactor"] = serde_json::json!([1.0, 1.0, 1.0]);
        }
        if let Some((occlusion_offset, occlusion_len)) = layout.occlusion_image_view {
            let texture_index = push_texture_image(occlusion_offset, occlusion_len);
            material["occlusionTexture"] = serde_json::json!({ "index": texture_index });
        }
        materials.push(material);
        primitive["material"] = serde_json::json!(material_index);
    }

    let mut gltf = serde_json::json!({
        "asset": {
            "version": "2.0",
            "generator": "bevy_synth_runtime"
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

fn pad_buffer_4(buffer: &mut Vec<u8>) {
    while !buffer.len().is_multiple_of(4) {
        buffer.push(0);
    }
}

fn encode_rgba_texture_png(
    texture: &SynthMeshTexture,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let expected = texture.width as usize * texture.height as usize * 4;
    if texture.rgba8.len() != expected {
        return Err(Box::new(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "texture byte length mismatch: expected {}, got {}",
                expected,
                texture.rgba8.len()
            ),
        )));
    }
    let mut out = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut out);
    encoder.write_image(
        texture.rgba8.as_slice(),
        texture.width,
        texture.height,
        image::ColorType::Rgba8.into(),
    )?;
    Ok(out)
}

pub fn resolve_output_path(
    output: Option<&PathBuf>,
    image_path: &Path,
    index: u32,
) -> Option<PathBuf> {
    let output = output?;
    if output.extension().is_none() || output.is_dir() {
        let dir = output.to_path_buf();
        let stem = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mesh");
        let suffix = if index == 0 {
            String::new()
        } else {
            format!("_{index}")
        };
        return Some(dir.join(format!("{stem}{suffix}.glb")));
    }

    let output = if output
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("glb"))
        .unwrap_or(false)
    {
        output.clone()
    } else {
        output.with_extension("glb")
    };

    if index == 0 {
        return Some(output);
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");
    let ext = "glb";
    Some(parent.join(format!("{stem}_{index}.{ext}")))
}

pub fn is_image_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tga" | "tif" | "tiff"
    )
}

pub fn is_mesh_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "glb" | "gltf" | "obj" | "fbx"
    )
}

pub fn load_text_embeds<B: Backend>(
    path: &Path,
    key: &str,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;
    let view = match safetensors.tensor(key) {
        Ok(tensor) => tensor,
        Err(_) => {
            let names = safetensors.names();
            if names.len() == 1 {
                safetensors.tensor(names[0])?
            } else {
                let available = names.join(", ");
                return Err(format!(
                    "text embeddings key '{key}' not found; available tensors: {available}"
                )
                .into());
            }
        }
    };

    let data = tensor_view_to_f32(&view)?;
    let shape = view.shape();
    let (batch, seq, dim) = match shape.len() {
        2 => (1, shape[0], shape[1]),
        3 => (shape[0], shape[1], shape[2]),
        _ => {
            return Err(format!(
                "expected text embeddings with rank 2 or 3, got shape {:?}",
                shape
            )
            .into());
        }
    };

    let tensor = Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([
        batch as i32,
        seq as i32,
        dim as i32,
    ]);
    Ok(tensor)
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use safetensors::Dtype;
    match view.dtype() {
        Dtype::F32 => {
            let data = bytemuck::cast_slice::<u8, f32>(view.data());
            Ok(data.to_vec())
        }
        Dtype::F16 => {
            let data = bytemuck::cast_slice::<u8, half::f16>(view.data());
            Ok(data.iter().map(|value| f32::from(*value)).collect())
        }
        Dtype::BF16 => {
            let data = bytemuck::cast_slice::<u8, half::bf16>(view.data());
            Ok(data.iter().map(|value| f32::from(*value)).collect())
        }
        other => Err(format!("unsupported text embedding dtype {other:?}").into()),
    }
}

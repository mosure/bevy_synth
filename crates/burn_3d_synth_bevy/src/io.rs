use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_3d_synth_tripo::pipeline::mesh::Mesh as TripoMesh;

pub(crate) fn write_obj(path: &Path, mesh: &TripoMesh) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let handle = fs::File::create(path)?;
    let mut writer = BufWriter::new(handle);
    for v in &mesh.vertices {
        writeln!(writer, "v {} {} {}", v[0], v[1], v[2])?;
    }
    for face in &mesh.faces {
        writeln!(writer, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)?;
    }
    writer.flush()?;
    Ok(())
}

pub(crate) fn resolve_output_path(
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
        return Some(dir.join(format!("{stem}{suffix}.obj")));
    }

    if index == 0 {
        return Some(output.clone());
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("mesh");
    let ext = output.extension().and_then(|s| s.to_str()).unwrap_or("obj");
    Some(parent.join(format!("{stem}_{index}.{ext}")))
}

pub(crate) fn is_image_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "bmp" | "gif" | "webp" | "tga" | "tif" | "tiff"
    )
}

pub(crate) fn is_mesh_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "glb" | "gltf" | "obj" | "fbx"
    )
}

pub(crate) fn load_text_embeds<B: Backend>(
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

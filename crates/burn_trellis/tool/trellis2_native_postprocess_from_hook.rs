#![recursion_limit = "256"]

use std::path::{Path, PathBuf};

use burn_trellis::hook_diff::{HookSnapshot, HookTensor};
use burn_trellis::staged_pipeline::{
    NativePbrPostprocessInput, native_pbr_mesh_from_decoded_tensors,
};
use clap::Parser;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(about = "Export native TRELLIS.2 PBR GLB from decoded hook tensors")]
struct Args {
    /// Safetensors hook containing decoded mesh and voxel tensors.
    #[arg(long)]
    hook: PathBuf,

    /// Output GLB path.
    #[arg(long)]
    output: PathBuf,

    /// Optional JSON report path.
    #[arg(long)]
    report_json: Option<PathBuf>,

    /// Optional target face count before native PBR/GLB export. Use 0 to disable.
    #[arg(long, default_value_t = 1_000_000)]
    target_faces: usize,

    /// Optional native PBR texture size. Use 0 for the runtime default.
    #[arg(long, default_value_t = 1024)]
    texture_size: usize,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let start = std::time::Instant::now();
    let hook = HookSnapshot::from_file(args.hook.as_path())
        .map_err(|err| format!("failed to load hook '{}': {err}", args.hook.display()))?;
    let hook_load_ms = start.elapsed().as_secs_f64() * 1000.0;

    let vertices = tensor_vec3(
        &hook,
        &[
            "decode_latent.mesh.0.vertices",
            "decode_shape_slat.meshes.0.vertices",
        ],
        &[
            "decode_latent.mesh.0.vertices_count",
            "decode_shape_slat.meshes.0.vertices_count",
        ],
    )?;
    let faces = tensor_faces(
        &hook,
        &[
            "decode_latent.mesh.0.faces",
            "decode_shape_slat.meshes.0.faces",
        ],
        &[
            "decode_latent.mesh.0.faces_count",
            "decode_shape_slat.meshes.0.faces_count",
        ],
    )?;
    let voxel_coords = tensor_coords4(
        &hook,
        &[
            "decode_tex_slat.voxels.coords",
            "decode_latent.mesh.0.voxel_coords",
        ],
        &["decode_latent.mesh.0.voxel_count"],
    )?;
    let voxel_attrs = tensor_rows6(
        &hook,
        &[
            "decode_tex_slat.voxels.feats",
            "decode_latent.mesh.0.voxel_attrs",
        ],
        &["decode_latent.mesh.0.voxel_count"],
    )?;
    let final_resolution = hook_scalar_u32(&hook, "run.final_resolution")
        .or_else(|| hook_spatial_resolution(&hook, "decode_tex_slat.voxels.spatial_shape"))
        .unwrap_or(512);

    let bake_start = std::time::Instant::now();
    let mesh = native_pbr_mesh_from_decoded_tensors(NativePbrPostprocessInput {
        vertices,
        faces,
        voxel_coords,
        voxel_attrs,
        final_resolution,
        target_faces: (args.target_faces > 0).then_some(args.target_faces),
        pbr_texture_size: (args.texture_size > 0).then_some(args.texture_size),
    })?;
    let bake_ms = bake_start.elapsed().as_secs_f64() * 1000.0;

    let write_start = std::time::Instant::now();
    burn_trellis::write_glb_mesh(args.output.as_path(), &mesh)?;
    let write_ms = write_start.elapsed().as_secs_f64() * 1000.0;

    if let Some(report_path) = args.report_json.as_ref() {
        write_report(
            report_path,
            json!({
                "status": "ok",
                "hook": args.hook,
                "output": args.output,
                "target_faces": args.target_faces,
                "texture_size": args.texture_size,
                "final_resolution": final_resolution,
                "mesh": {
                    "vertices": mesh.vertices.len(),
                    "faces": mesh.faces.len(),
                    "uvs": mesh.uvs.len(),
                    "has_pbr_textures": mesh.has_pbr_textures(),
                    "texture_resolution": mesh.pbr_resolution(),
                    "texture_texels": mesh.texture_texel_count(),
                },
                "timings_ms": {
                    "hook_load": hook_load_ms,
                    "native_pbr": bake_ms,
                    "write_glb": write_ms,
                    "total": hook_load_ms + bake_ms + write_ms,
                },
            }),
        )?;
    }
    Ok(())
}

fn write_report(path: &Path, payload: serde_json::Value) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create '{}': {err}", parent.display()))?;
    }
    std::fs::write(
        path,
        serde_json::to_string_pretty(&payload)
            .map_err(|err| format!("failed to encode report json: {err}"))?
            + "\n",
    )
    .map_err(|err| format!("failed to write '{}': {err}", path.display()))
}

fn first_tensor<'a>(
    hook: &'a HookSnapshot,
    keys: &'a [&'a str],
) -> Result<(&'a str, &'a HookTensor), String> {
    for key in keys {
        if let Some(tensor) = hook.tensors.get(*key) {
            return Ok((*key, tensor));
        }
    }
    Err(format!(
        "hook is missing all required keys: {}",
        keys.join(", ")
    ))
}

fn tensor_count(hook: &HookSnapshot, keys: &[&str], fallback: usize) -> Result<usize, String> {
    for key in keys {
        if let Some(tensor) = hook.tensors.get(*key) {
            if tensor.data.len() != 1 {
                return Err(format!(
                    "hook tensor '{key}' has invalid count length {}; expected 1",
                    tensor.data.len()
                ));
            }
            return Ok(f32_to_usize_count(key, tensor.data[0])?.min(fallback));
        }
    }
    Ok(fallback)
}

fn tensor_vec3(
    hook: &HookSnapshot,
    keys: &[&str],
    count_keys: &[&str],
) -> Result<Vec<[f32; 3]>, String> {
    let (key, tensor) = first_tensor(hook, keys)?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 3 {
        return Err(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 3]",
            tensor.shape
        ));
    }
    let rows = tensor_count(hook, count_keys, tensor.shape[0])?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 3;
        out.push([
            tensor.data[base],
            tensor.data[base + 1],
            tensor.data[base + 2],
        ]);
    }
    Ok(out)
}

fn tensor_faces(
    hook: &HookSnapshot,
    keys: &[&str],
    count_keys: &[&str],
) -> Result<Vec<[u32; 3]>, String> {
    let (key, tensor) = first_tensor(hook, keys)?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 3 {
        return Err(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 3]",
            tensor.shape
        ));
    }
    let rows = tensor_count(hook, count_keys, tensor.shape[0])?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 3;
        out.push([
            f32_to_u32_index(key, tensor.data[base])?,
            f32_to_u32_index(key, tensor.data[base + 1])?,
            f32_to_u32_index(key, tensor.data[base + 2])?,
        ]);
    }
    Ok(out)
}

fn tensor_coords4(
    hook: &HookSnapshot,
    keys: &[&str],
    count_keys: &[&str],
) -> Result<Vec<[u32; 4]>, String> {
    let (key, tensor) = first_tensor(hook, keys)?;
    if tensor.shape.len() != 2 || (tensor.shape[1] != 3 && tensor.shape[1] != 4) {
        return Err(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 3] or [N, 4]",
            tensor.shape
        ));
    }
    let cols = tensor.shape[1];
    let rows = tensor_count(hook, count_keys, tensor.shape[0])?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * cols;
        if cols == 4 {
            out.push([
                f32_to_u32_index(key, tensor.data[base])?,
                f32_to_u32_index(key, tensor.data[base + 1])?,
                f32_to_u32_index(key, tensor.data[base + 2])?,
                f32_to_u32_index(key, tensor.data[base + 3])?,
            ]);
        } else {
            out.push([
                0,
                f32_to_u32_index(key, tensor.data[base])?,
                f32_to_u32_index(key, tensor.data[base + 1])?,
                f32_to_u32_index(key, tensor.data[base + 2])?,
            ]);
        }
    }
    Ok(out)
}

fn tensor_rows6(
    hook: &HookSnapshot,
    keys: &[&str],
    count_keys: &[&str],
) -> Result<Vec<[f32; 6]>, String> {
    let (key, tensor) = first_tensor(hook, keys)?;
    if tensor.shape.len() != 2 || tensor.shape[1] != 6 {
        return Err(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 6]",
            tensor.shape
        ));
    }
    let rows = tensor_count(hook, count_keys, tensor.shape[0])?;
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 6;
        out.push([
            tensor.data[base],
            tensor.data[base + 1],
            tensor.data[base + 2],
            tensor.data[base + 3],
            tensor.data[base + 4],
            tensor.data[base + 5],
        ]);
    }
    Ok(out)
}

fn hook_scalar_u32(hook: &HookSnapshot, key: &str) -> Option<u32> {
    hook.tensors
        .get(key)
        .and_then(|tensor| tensor.data.first())
        .and_then(|value| f32_to_u32_index(key, *value).ok())
}

fn hook_spatial_resolution(hook: &HookSnapshot, key: &str) -> Option<u32> {
    hook.tensors.get(key).and_then(|tensor| {
        tensor
            .data
            .iter()
            .filter_map(|value| f32_to_u32_index(key, *value).ok())
            .max()
    })
}

fn f32_to_usize_count(label: &str, value: f32) -> Result<usize, String> {
    if !value.is_finite() || value < 0.0 || (value.round() - value).abs() > 1.0e-3 {
        return Err(format!(
            "hook tensor '{label}' contains invalid count value {value}"
        ));
    }
    Ok(value.round() as usize)
}

fn f32_to_u32_index(label: &str, value: f32) -> Result<u32, String> {
    if !value.is_finite() || value < 0.0 || (value.round() - value).abs() > 1.0e-3 {
        return Err(format!(
            "hook tensor '{label}' contains invalid integer value {value}"
        ));
    }
    let rounded = value.round();
    if rounded > u32::MAX as f32 {
        return Err(format!(
            "hook tensor '{label}' contains out-of-range integer value {value}"
        ));
    }
    Ok(rounded as u32)
}

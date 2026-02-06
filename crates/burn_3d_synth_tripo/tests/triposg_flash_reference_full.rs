#![cfg(feature = "import")]
#![recursion_limit = "256"]

use std::fs;
use std::path::{Path, PathBuf};

use burn::prelude::*;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use burn_3d_synth_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_3d_synth_tripo::model::triposg::vae::import::load_triposg_vae;
use burn_3d_synth_tripo::pipeline::geometry::{FlashExtractConfig, flash_extract_geometry};
use burn_3d_synth_tripo::pipeline::mesh::{DenseGrid, Mesh as TripoMesh, sdf_to_mesh_diff_dmc};

type GpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";

#[test]
fn triposg_flash_full_reference_matches() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("TRIPOSG_FULL_REFERENCE").is_err() {
        eprintln!("skipping: set TRIPOSG_FULL_REFERENCE=1 to run full flash reference test");
        return Ok(());
    }
    if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
        eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run full flash reference test");
        return Ok(());
    }
    if !wgpu_available() {
        eprintln!("skipping: wgpu backend not available on this system");
        return Ok(());
    }

    let reference_path = asset_path("assets/hooks/triposg_pipeline_reference_full.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: full reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_root = resolve_weights_root();
    if !weights_root.exists() {
        eprintln!(
            "skipping: TripoSG weights root not found at {}",
            weights_root.display()
        );
        return Ok(());
    }

    let bytes = fs::read(&reference_path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;

    let latents = read_f32_tensor(&safetensors, "output.latents")?;
    let latents_shape = latents.shape.clone();
    let bounds = read_f32_vec(&safetensors, "meta.bounds")?;
    let bounds = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ];

    let flash_octree_depth = read_scalar_i32(&safetensors, "meta.flash_octree_depth")? as usize;
    let flash_min_resolution =
        read_scalar_i32(&safetensors, "meta.flash_min_resolution")? as usize;
    let flash_mini_grid_num =
        read_scalar_i32(&safetensors, "meta.flash_mini_grid_num")? as usize;
    let flash_num_chunks = read_scalar_i32(&safetensors, "meta.flash_num_chunks")? as usize;
    let flash_mc_level = read_scalar_f32(&safetensors, "meta.flash_mc_level")?;
    let grid_size = read_scalar_i32(&safetensors, "output.grid.size")? as usize;

    let sample_indices = read_i32_tensor(&safetensors, "output.grid.sample_indices")?;
    let sample_values = read_f32_vec(&safetensors, "output.grid.sample_values")?;

    let device = burn_wgpu::WgpuDevice::default();
    let vae_config = TripoSGVaeConfig::from_config_file(
        weights_root.join("vae/config.json"),
    )
    .unwrap_or_else(|_| TripoSGVaeConfig::midi_3d());
    let vae = load_triposg_vae::<GpuBackend>(
        &vae_config,
        &device,
        weights_root.join("vae/diffusion_pytorch_model.safetensors"),
    )?;

    let latents_tensor = tensor_from_vec::<GpuBackend, 3>(
        latents.data.clone(),
        [
            latents_shape[0],
            latents_shape[1],
            latents_shape[2],
        ],
        &device,
    );

    let flash = FlashExtractConfig {
        bounds,
        octree_depth: flash_octree_depth,
        num_chunks: flash_num_chunks,
        mc_level: flash_mc_level,
        min_resolution: flash_min_resolution,
        mini_grid_num: flash_mini_grid_num,
    };

    unsafe {
        std::env::set_var("TRIPOSG_FLASH_NO_FALLBACK", "1");
    }
    let grid = flash_extract_geometry(latents_tensor, &vae, &flash)?;

    assert_eq!(
        grid.size,
        [grid_size, grid_size, grid_size],
        "grid size mismatch"
    );
    compare_sampled_grid(&grid, &sample_indices, &sample_values)?;

    let reference_mesh = read_reference_mesh(&safetensors)?;
    let mesh = sdf_to_mesh_diff_dmc(&grid);
    compare_mesh_bounds(&mesh, &reference_mesh)?;

    Ok(())
}

#[test]
fn triposg_flash_samples_match_reference() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("TRIPOSG_FULL_REFERENCE").is_err() {
        eprintln!("skipping: set TRIPOSG_FULL_REFERENCE=1 to run flash sample reference test");
        return Ok(());
    }
    if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
        eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run flash sample reference test");
        return Ok(());
    }
    if !wgpu_available() {
        eprintln!("skipping: wgpu backend not available on this system");
        return Ok(());
    }

    let reference_path = asset_path("assets/hooks/triposg_pipeline_reference_full.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: full reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_root = resolve_weights_root();
    if !weights_root.exists() {
        eprintln!(
            "skipping: TripoSG weights root not found at {}",
            weights_root.display()
        );
        return Ok(());
    }

    let bytes = fs::read(&reference_path)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;

    let latents = read_f32_tensor(&safetensors, "output.latents")?;
    let bounds = read_f32_vec(&safetensors, "meta.bounds")?;
    let bounds = [
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    ];
    let flash_octree_depth = read_scalar_i32(&safetensors, "meta.flash_octree_depth")? as usize;
    let grid_size = read_scalar_i32(&safetensors, "output.grid.size")? as usize;

    let sample_indices = read_i32_tensor(&safetensors, "output.grid.sample_indices")?;
    let sample_values = read_f32_vec(&safetensors, "output.grid.sample_values")?;

    let device = burn_wgpu::WgpuDevice::default();
    let vae_config = TripoSGVaeConfig::from_config_file(
        weights_root.join("vae/config.json"),
    )
    .unwrap_or_else(|_| TripoSGVaeConfig::midi_3d());
    let vae = load_triposg_vae::<GpuBackend>(
        &vae_config,
        &device,
        weights_root.join("vae/diffusion_pytorch_model.safetensors"),
    )?;

    let latents_tensor = tensor_from_vec::<GpuBackend, 3>(
        latents.data.clone(),
        [
            latents.shape[0],
            latents.shape[1],
            latents.shape[2],
        ],
        &device,
    );

    let coords = flash_sample_coords(bounds, grid_size, &sample_indices);
    let coords_tensor = tensor_from_vec::<GpuBackend, 3>(
        coords,
        [1, (sample_indices.data.len() / 3) as usize, 3],
        &device,
    );

    let decoded = vae.decode(coords_tensor.clone(), latents_tensor.clone(), None);
    let decoded_values = decoded
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to read decoded flash samples")?;

    let latent_proj = vae.prepare_latent_projection(latents_tensor, None);
    let kv_cache = vae.build_kv_cache(latent_proj.clone(), None);
    let (decoded_cached, _) =
        vae.decode_with_latent_projection(coords_tensor, latent_proj, Some(kv_cache), None);
    let decoded_cached_values = decoded_cached
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|_| "failed to read cached decoded flash samples")?;
    compare_sampled_logits(&decoded_values, &decoded_cached_values)?;

    compare_sampled_sdf(
        &decoded_values,
        &sample_values,
        flash_octree_depth,
    )?;

    Ok(())
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

fn resolve_weights_root() -> PathBuf {
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let path = PathBuf::from(root);
        if path.exists() {
            return path;
        }
    }
    let path = PathBuf::from(TRIPOSG_ROOT);
    if path.exists() {
        return path;
    }
    asset_path("assets/models/MIDI-3D")
}

#[derive(Clone)]
struct HookTensorF32 {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct HookTensorI32 {
    shape: Vec<usize>,
    data: Vec<i32>,
}

fn read_f32_tensor(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HookTensorF32, Box<dyn std::error::Error>> {
    let view = safetensors.tensor(name)?;
    if view.dtype() != Dtype::F32 {
        return Err(format!("expected f32 tensor for {name}").into());
    }
    Ok(HookTensorF32 {
        shape: view.shape().to_vec(),
        data: tensor_view_to_vec_f32(&view),
    })
}

fn read_i32_tensor(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> Result<HookTensorI32, Box<dyn std::error::Error>> {
    let view = safetensors.tensor(name)?;
    if view.dtype() != Dtype::I32 {
        return Err(format!("expected i32 tensor for {name}").into());
    }
    Ok(HookTensorI32 {
        shape: view.shape().to_vec(),
        data: tensor_view_to_vec_i32(&view),
    })
}

fn read_scalar_f32(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let view = safetensors.tensor(name)?;
    if view.dtype() != Dtype::F32 {
        return Err(format!("expected f32 scalar for {name}").into());
    }
    Ok(tensor_view_to_vec_f32(&view)
        .first()
        .copied()
        .ok_or("missing scalar value")?)
}

fn read_scalar_i32(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> Result<i32, Box<dyn std::error::Error>> {
    let view = safetensors.tensor(name)?;
    match view.dtype() {
        Dtype::F32 => Ok(tensor_view_to_vec_f32(&view)
            .first()
            .copied()
            .ok_or("missing scalar value")? as i32),
        Dtype::I32 => Ok(tensor_view_to_vec_i32(&view)
            .first()
            .copied()
            .ok_or("missing scalar value")?),
        _ => Err(format!("unexpected scalar dtype for {name}").into()),
    }
}

fn read_f32_vec(
    safetensors: &SafeTensors<'_>,
    name: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let view = safetensors.tensor(name)?;
    if view.dtype() != Dtype::F32 {
        return Err(format!("expected f32 vector for {name}").into());
    }
    Ok(tensor_view_to_vec_f32(&view))
}

fn read_reference_mesh(
    safetensors: &SafeTensors<'_>,
) -> Result<Option<TripoMesh>, Box<dyn std::error::Error>> {
    let verts = read_f32_tensor(safetensors, "output.mesh.vertices")?;
    let faces = read_i32_tensor(safetensors, "output.mesh.faces")?;
    if verts.data.is_empty() || faces.data.is_empty() {
        return Ok(None);
    }
    let mut vertices = Vec::with_capacity(verts.data.len() / 3);
    for chunk in verts.data.chunks_exact(3) {
        vertices.push([chunk[0], chunk[1], chunk[2]]);
    }
    let mut mesh_faces = Vec::with_capacity(faces.data.len() / 3);
    for chunk in faces.data.chunks_exact(3) {
        mesh_faces.push([chunk[0] as u32, chunk[1] as u32, chunk[2] as u32]);
    }
    Ok(Some(TripoMesh {
        vertices,
        faces: mesh_faces,
    }))
}

fn tensor_view_to_vec_f32(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn tensor_view_to_vec_i32(view: &TensorView<'_>) -> Vec<i32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            i32::from_le_bytes(bytes)
        })
        .collect()
}

fn tensor_from_vec<B: Backend, const D: usize>(
    data: Vec<f32>,
    shape: [usize; D],
    device: &B::Device,
) -> Tensor<B, D> {
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    let shape_i32 = shape.map(|v| v as i32);
    flat.reshape(shape_i32)
}

fn compare_sampled_grid(
    grid: &DenseGrid,
    indices: &HookTensorI32,
    values: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let expected = indices.data.len() / 3;
    if values.len() != expected {
        return Err(format!(
            "sample values length mismatch: {} vs {}",
            values.len(),
            expected
        )
        .into());
    }
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut mse = 0.0f32;
    let mut count = 0usize;

    for (idx, &ref_value) in indices.data.chunks_exact(3).zip(values.iter()) {
        let x = idx[0] as usize;
        let y = idx[1] as usize;
        let z = idx[2] as usize;
        let grid_idx = (z * grid.size[1] + y) * grid.size[0] + x;
        let value = grid.values[grid_idx];
        if value.is_nan() && ref_value.is_nan() {
            continue;
        }
        let diff = value - ref_value;
        let abs = diff.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        mse += diff * diff;
        count += 1;
    }

    let count = count.max(1) as f32;
    let mean_abs = sum_abs / count;
    let mse = mse / count;

    let max_tol = 0.2;
    let mean_tol = 0.03;
    let mse_tol = 0.05;
    if max_abs > max_tol || mean_abs > mean_tol || mse > mse_tol {
        return Err(format!(
            "flash grid samples out of tolerance: mean_abs={mean_abs:.6} max_abs={max_abs:.6} mse={mse:.6}"
        )
        .into());
    }

    Ok(())
}

fn flash_sample_coords(
    bounds: [f32; 6],
    grid_size: usize,
    indices: &HookTensorI32,
) -> Vec<f32> {
    let resolution = (grid_size - 1).max(1) as f32;
    let step_x = (bounds[3] - bounds[0]) / resolution;
    let step_y = (bounds[4] - bounds[1]) / resolution;
    let step_z = (bounds[5] - bounds[2]) / resolution;
    let mut coords = Vec::with_capacity(indices.data.len());
    for idx in indices.data.chunks_exact(3) {
        let x = idx[0] as f32;
        let y = idx[1] as f32;
        let z = idx[2] as f32;
        coords.push(bounds[0] + step_x * x);
        coords.push(bounds[1] + step_y * y);
        coords.push(bounds[2] + step_z * z);
    }
    coords
}

fn compare_sampled_sdf(
    decoded: &[f32],
    reference: &[f32],
    octree_depth: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = reference.len().min(decoded.len());
    let scale = -1.0 / (1usize << octree_depth) as f32;

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut mse = 0.0f32;
    let mut count = 0usize;

    for idx in 0..samples {
        let ref_value = reference[idx];
        if ref_value.is_nan() {
            continue;
        }
        let sdf = decoded[idx] * scale;
        let diff = sdf - ref_value;
        let abs = diff.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        mse += diff * diff;
        count += 1;
    }

    let count = count.max(1) as f32;
    let mean_abs = sum_abs / count;
    let mse = mse / count;

    let max_tol = 0.2;
    let mean_tol = 0.03;
    let mse_tol = 0.05;
    if max_abs > max_tol || mean_abs > mean_tol || mse > mse_tol {
        return Err(format!(
            "flash sample decode out of tolerance: mean_abs={mean_abs:.6} max_abs={max_abs:.6} mse={mse:.6}"
        )
        .into());
    }

    Ok(())
}

fn compare_sampled_logits(
    direct: &[f32],
    cached: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if direct.len() != cached.len() {
        return Err("cached decode length mismatch".into());
    }

    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut count = 0usize;

    for (&a, &b) in direct.iter().zip(cached.iter()) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
        count += 1;
    }

    let mean_abs = sum_abs / count.max(1) as f32;
    let max_tol = 1e-3;
    let mean_tol = 1e-4;
    if max_abs > max_tol || mean_abs > mean_tol {
        return Err(format!(
            "cached decode mismatch: mean_abs={mean_abs:.6} max_abs={max_abs:.6}"
        )
        .into());
    }

    Ok(())
}

fn compare_mesh_bounds(
    mesh: &Option<TripoMesh>,
    reference: &Option<TripoMesh>,
) -> Result<(), Box<dyn std::error::Error>> {
    match (mesh, reference) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => {
            Err("mesh extraction mismatch: reference/actual presence differs".into())
        }
        (Some(mesh), Some(reference)) => {
            let (min_a, max_a) = mesh_bounds(mesh);
            let (min_b, max_b) = mesh_bounds(reference);
            let tol = 0.05;
            for axis in 0..3 {
                let min_diff = (min_a[axis] - min_b[axis]).abs();
                let max_diff = (max_a[axis] - max_b[axis]).abs();
                if min_diff > tol || max_diff > tol {
                    return Err(format!(
                        "mesh bounds mismatch on axis {axis}: [{}, {}] vs [{}, {}]",
                        min_a[axis], max_a[axis], min_b[axis], max_b[axis]
                    )
                    .into());
                }
            }
            Ok(())
        }
    }
}

fn mesh_bounds(mesh: &TripoMesh) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }
    (min, max)
}

fn wgpu_available() -> bool {
    std::panic::catch_unwind(|| {
        let _device = burn_wgpu::WgpuDevice::default();
    })
    .is_ok()
}

#![cfg(feature = "import")]
#![recursion_limit = "256"]

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use burn::prelude::*;
use burn_store::{BurnpackStore, ModuleSnapshot};

use burn_tripo::model::triposg::{
    dit::{TripoSGDiT, TripoSGDiTConfig},
    vae::{TripoSGVae, TripoSGVaeConfig},
};
use burn_tripo::pipeline::geometry::{FlashExtractConfig, flash_extract_geometry};

type CpuBackend = burn::backend::NdArray<f32>;
type GpuBackend = burn_wgpu::Wgpu<f32, i32, u32>;

struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }

    fn unset(key: &'static str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe {
            std::env::remove_var(key);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            if let Some(value) = &self.previous {
                std::env::set_var(self.key, value);
            } else {
                std::env::remove_var(self.key);
            }
        }
    }
}

fn temp_burnpack_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.push(format!("burn_synth_{label}_{stamp}.bpk"));
    path
}

fn make_data(len: usize, scale: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(len);
    for idx in 0..len {
        let x = idx as f32 * 0.013;
        data.push((x.sin() * 0.75 + x.cos() * 0.25) * scale);
    }
    data
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

fn tensor_to_vec<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    label: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|err| format!("{label}: failed to read tensor data: {err:?}").into())
}

fn compare_vectors(label: &str, cpu: &[f32], gpu: &[f32], tol_max: f32, tol_mean: f32) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: length mismatch");
    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut count = 0usize;
    for (&a, &b) in cpu.iter().zip(gpu.iter()) {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        count += 1;
    }
    let mean = sum_diff / count.max(1) as f32;
    assert!(
        max_diff <= tol_max,
        "{label}: max diff {max_diff:.4} > {tol_max:.4}"
    );
    assert!(
        mean <= tol_mean,
        "{label}: mean diff {mean:.4} > {tol_mean:.4}"
    );
}

fn compare_grids(
    cpu: &burn_tripo::pipeline::mesh::DenseGrid,
    gpu: &burn_tripo::pipeline::mesh::DenseGrid,
    tol_max: f32,
    tol_mean: f32,
    max_nan_mismatch: usize,
) {
    assert_eq!(cpu.size, gpu.size, "grid size mismatch");
    assert_eq!(cpu.bounds, gpu.bounds, "grid bounds mismatch");

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut count = 0usize;
    let mut nan_mismatch = 0usize;

    for (&a, &b) in cpu.values.iter().zip(gpu.values.iter()) {
        let a_nan = a.is_nan();
        let b_nan = b.is_nan();
        if a_nan || b_nan {
            if a_nan != b_nan {
                nan_mismatch += 1;
            }
            continue;
        }
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        sum_diff += diff;
        count += 1;
    }

    assert!(
        nan_mismatch <= max_nan_mismatch,
        "grid NaN mismatch count {nan_mismatch} > {max_nan_mismatch}"
    );
    let mean = sum_diff / count.max(1) as f32;
    assert!(
        max_diff <= tol_max,
        "grid max diff {max_diff:.4} > {tol_max:.4}"
    );
    assert!(mean <= tol_mean, "grid mean diff {mean:.4} > {tol_mean:.4}");
}

fn load_same_weights<M: Module<CpuBackend>>(
    model: &M,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_file(path).overwrite(true);
    model.save_into(&mut store)?;
    Ok(())
}

fn load_weights_into<M: Module<B>, B: Backend>(
    model: &mut M,
    path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_file(path).validate(true);
    model.load_from(&mut store)?;
    Ok(())
}

fn wgpu_available() -> bool {
    std::panic::catch_unwind(|| {
        let _device = burn_wgpu::WgpuDevice::default();
    })
    .is_ok()
}

#[test]
fn gpu_dit_matches_cpu_small() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
        eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run gpu correctness tests");
        return Ok(());
    }
    if !wgpu_available() {
        eprintln!("skipping: wgpu backend not available on this system");
        return Ok(());
    }

    let cpu_device = <CpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let gpu_device = <GpuBackend as burn::tensor::backend::BackendTypes>::Device::default();

    let config = TripoSGDiTConfig {
        in_channels: 8,
        width: 32,
        num_layers: 2,
        num_attention_heads: 4,
        cross_attention_dim: 16,
        cross_attention_2_dim: None,
    };

    let cpu_model = TripoSGDiT::<CpuBackend>::new(&cpu_device, config.clone());
    let burnpack_path = temp_burnpack_path("dit_small");
    load_same_weights(&cpu_model, &burnpack_path)?;

    let mut gpu_model = TripoSGDiT::<GpuBackend>::new(&gpu_device, config);
    load_weights_into(&mut gpu_model, &burnpack_path)?;

    let hidden = tensor_from_vec::<CpuBackend, 3>(make_data(4 * 8, 0.7), [1, 4, 8], &cpu_device);
    let timestep = tensor_from_vec::<CpuBackend, 1>(make_data(1, 0.1), [1], &cpu_device);
    let encoder = tensor_from_vec::<CpuBackend, 3>(make_data(3 * 16, 0.5), [1, 3, 16], &cpu_device);

    let hidden_gpu = tensor_from_vec::<GpuBackend, 3>(
        tensor_to_vec(hidden.clone(), "dit.hidden")?,
        [1, 4, 8],
        &gpu_device,
    );
    let timestep_gpu = tensor_from_vec::<GpuBackend, 1>(
        tensor_to_vec(timestep.clone(), "dit.timestep")?,
        [1],
        &gpu_device,
    );
    let encoder_gpu = tensor_from_vec::<GpuBackend, 3>(
        tensor_to_vec(encoder.clone(), "dit.encoder")?,
        [1, 3, 16],
        &gpu_device,
    );

    let cpu_out = cpu_model.forward(hidden, timestep, encoder, None, None);
    let gpu_out = gpu_model.forward(hidden_gpu, timestep_gpu, encoder_gpu, None, None);

    let cpu_vec = tensor_to_vec(cpu_out, "dit.output.cpu")?;
    let gpu_vec = tensor_to_vec(gpu_out, "dit.output.gpu")?;
    compare_vectors("dit_small", &cpu_vec, &gpu_vec, 5e-2, 1e-2);

    let _ = std::fs::remove_file(burnpack_path);
    Ok(())
}

#[test]
fn gpu_vae_decode_matches_cpu_small() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
        eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run gpu correctness tests");
        return Ok(());
    }
    if !wgpu_available() {
        eprintln!("skipping: wgpu backend not available on this system");
        return Ok(());
    }

    let cpu_device = <CpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let gpu_device = <GpuBackend as burn::tensor::backend::BackendTypes>::Device::default();

    let config = TripoSGVaeConfig {
        embed_frequency: 2,
        embed_include_pi: false,
        embedding_type: "frequency".to_string(),
        in_channels: 3,
        latent_channels: 8,
        num_attention_heads: 4,
        num_layers_decoder: 2,
        num_layers_encoder: 2,
        width_decoder: 32,
        width_encoder: 32,
    };

    let cpu_model = TripoSGVae::<CpuBackend>::new(&cpu_device, config.clone());
    let burnpack_path = temp_burnpack_path("vae_small");
    load_same_weights(&cpu_model, &burnpack_path)?;

    let mut gpu_model = TripoSGVae::<GpuBackend>::new(&gpu_device, config);
    load_weights_into(&mut gpu_model, &burnpack_path)?;

    let latents = tensor_from_vec::<CpuBackend, 3>(make_data(4 * 8, 0.9), [1, 4, 8], &cpu_device);
    let coords = tensor_from_vec::<CpuBackend, 3>(make_data(6 * 3, 0.2), [1, 6, 3], &cpu_device);

    let latents_gpu = tensor_from_vec::<GpuBackend, 3>(
        tensor_to_vec(latents.clone(), "vae.latents")?,
        [1, 4, 8],
        &gpu_device,
    );
    let coords_gpu = tensor_from_vec::<GpuBackend, 3>(
        tensor_to_vec(coords.clone(), "vae.coords")?,
        [1, 6, 3],
        &gpu_device,
    );

    let cpu_out = cpu_model.decode(coords, &latents, None);
    let gpu_out = gpu_model.decode(coords_gpu, &latents_gpu, None);

    let cpu_vec = tensor_to_vec(cpu_out, "vae.output.cpu")?;
    let gpu_vec = tensor_to_vec(gpu_out, "vae.output.gpu")?;
    compare_vectors("vae_decode_small", &cpu_vec, &gpu_vec, 5e-2, 1e-2);

    let _ = std::fs::remove_file(burnpack_path);
    Ok(())
}

#[test]
fn gpu_flash_extract_matches_cpu_small() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var("BURN_WGPU_CORRECTNESS").is_err() {
        eprintln!("skipping: set BURN_WGPU_CORRECTNESS=1 to run gpu correctness tests");
        return Ok(());
    }
    if !wgpu_available() {
        eprintln!("skipping: wgpu backend not available on this system");
        return Ok(());
    }

    let cpu_device = <CpuBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let gpu_device = <GpuBackend as burn::tensor::backend::BackendTypes>::Device::default();

    let config = TripoSGVaeConfig {
        embed_frequency: 2,
        embed_include_pi: false,
        embedding_type: "frequency".to_string(),
        in_channels: 3,
        latent_channels: 8,
        num_attention_heads: 4,
        num_layers_decoder: 2,
        num_layers_encoder: 2,
        width_decoder: 32,
        width_encoder: 32,
    };

    let cpu_model = TripoSGVae::<CpuBackend>::new(&cpu_device, config.clone());
    let burnpack_path = temp_burnpack_path("vae_flash_small");
    load_same_weights(&cpu_model, &burnpack_path)?;

    let mut gpu_model = TripoSGVae::<GpuBackend>::new(&gpu_device, config);
    load_weights_into(&mut gpu_model, &burnpack_path)?;

    let latents = tensor_from_vec::<CpuBackend, 3>(make_data(4 * 8, 0.75), [1, 4, 8], &cpu_device);
    let latents_gpu = tensor_from_vec::<GpuBackend, 3>(
        tensor_to_vec(latents.clone(), "flash.latents")?,
        [1, 4, 8],
        &gpu_device,
    );

    let flash = FlashExtractConfig {
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        octree_depth: 3,
        num_chunks: 128,
        mc_level: 0.0,
        min_resolution: 3,
        mini_grid_num: 1,
    };

    let _guard_cpu = EnvVarGuard::unset("TRIPOSG_FLASH_CPU");
    let _guard_no_fallback = EnvVarGuard::set("TRIPOSG_FLASH_NO_FALLBACK", "1");
    let cpu_grid = flash_extract_geometry(&latents, &cpu_model, &flash)?;
    let gpu_grid = flash_extract_geometry(&latents_gpu, &gpu_model, &flash)?;

    assert!(
        cpu_grid.values.iter().any(|value| value.is_finite()),
        "cpu flash grid is all NaNs"
    );
    assert!(
        gpu_grid.values.iter().any(|value| value.is_finite()),
        "gpu flash grid is all NaNs"
    );
    let max_nan_mismatch = (cpu_grid.values.len() / 200).max(1);
    compare_grids(&cpu_grid, &gpu_grid, 0.1, 0.02, max_nan_mismatch);

    let _ = std::fs::remove_file(burnpack_path);
    Ok(())
}

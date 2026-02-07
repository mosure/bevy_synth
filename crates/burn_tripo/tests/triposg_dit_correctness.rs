#![cfg(feature = "import")]

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_tripo::model::triposg::{
    dit::{TripoSGDiTConfig, import::load_triposg_dit},
    hooks::HookRecorder,
};

const FALLBACK_WEIGHTS_PATH: &str =
    "assets/models/MIDI-3D/transformer/diffusion_pytorch_model.safetensors";
const TRIPOSG_ROOT: &str = r"E:\repos\TripoSG\pretrained_weights\TripoSG";

const MAX_ABS: f32 = 5e-2;
const MEAN_ABS: f32 = 5e-3;
const MSE: f32 = 5e-3;

#[test]
fn triposg_dit_hooks_match_reference() -> Result<(), Box<dyn std::error::Error>> {
    let reference_path = asset_path("assets/hooks/triposg_dit_reference.safetensors");
    if !reference_path.exists() {
        eprintln!(
            "skipping: hook reference file not found at {}",
            reference_path.display()
        );
        return Ok(());
    }

    let weights_path = resolve_weights_path("transformer/diffusion_pytorch_model.safetensors");
    if !weights_path.exists() {
        eprintln!(
            "skipping: DiT weights file not found at {}",
            weights_path.display()
        );
        return Ok(());
    }

    let reference = HookReference::load(reference_path.as_path())?;
    let device = Default::default();
    let config = weights_path
        .parent()
        .and_then(|path| TripoSGDiTConfig::from_config_file(path.join("config.json")).ok())
        .unwrap_or_else(TripoSGDiTConfig::midi_3d);
    let model = load_triposg_dit::<burn::backend::NdArray<f32>>(&config, &device, &weights_path)?;

    let hidden_states = reference
        .get_input("input.hidden_states")
        .ok_or("missing input.hidden_states in reference")?;
    let encoder_hidden_states = reference
        .get_input("input.encoder_hidden_states")
        .ok_or("missing input.encoder_hidden_states in reference")?;
    let encoder_hidden_states_2 = reference
        .get_input("input.encoder_hidden_states_2")
        .ok_or("missing input.encoder_hidden_states_2 in reference")?;
    let timestep = reference
        .get_input("input.timestep")
        .ok_or("missing input.timestep in reference")?;

    let hidden_states =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&hidden_states, &device)?;
    let encoder_hidden_states =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&encoder_hidden_states, &device)?;
    let encoder_hidden_states_2 =
        tensor_from_data_3d::<burn::backend::NdArray<f32>>(&encoder_hidden_states_2, &device)?;
    let timestep = tensor_from_data_1d::<burn::backend::NdArray<f32>>(&timestep, &device)?;

    let mut hooks = HookRecorder::new();
    let _output = model.forward(
        hidden_states,
        timestep,
        encoder_hidden_states,
        Some(encoder_hidden_states_2),
        Some(&mut hooks),
    );

    for (name, reference_tensor) in reference.tensors.iter() {
        if name.starts_with("input.") {
            continue;
        }
        let Some(burn_tensor) = hooks.tensors().get(name) else {
            return Err(format!("missing hook tensor `{name}` in burn output").into());
        };
        if burn_tensor.shape != reference_tensor.shape {
            return Err(format!(
                "shape mismatch for `{name}`: burn {:?} vs ref {:?}",
                burn_tensor.shape, reference_tensor.shape
            )
            .into());
        }

        let stats = compute_stats(&burn_tensor.data, &reference_tensor.data);
        if stats.max_abs > MAX_ABS || stats.mean_abs > MEAN_ABS || stats.mse > MSE {
            return Err(format!(
                "hook `{name}` out of tolerance: mean_abs={:.6} max_abs={:.6} mse={:.6}",
                stats.mean_abs, stats.max_abs, stats.mse
            )
            .into());
        }
    }

    Ok(())
}

fn resolve_weights_path(leaf: &str) -> std::path::PathBuf {
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let candidate = Path::new(&root).join(leaf);
        if candidate.exists() {
            return candidate;
        }
    }
    let candidate = Path::new(TRIPOSG_ROOT).join(leaf);
    if candidate.exists() {
        return candidate;
    }
    asset_path(FALLBACK_WEIGHTS_PATH)
}

fn asset_path(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)
}

struct HookReference {
    tensors: BTreeMap<String, HookTensor>,
}

#[derive(Clone)]
struct HookTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl HookReference {
    fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = fs::read(path)?;
        let safetensors = SafeTensors::deserialize(&bytes)?;
        let mut tensors = BTreeMap::new();
        for name in safetensors.names() {
            let view = safetensors.tensor(name)?;
            let data = tensor_view_to_vec(&view);
            tensors.insert(
                name.to_string(),
                HookTensor {
                    shape: view.shape().to_vec(),
                    data,
                },
            );
        }
        Ok(Self { tensors })
    }

    fn get_input(&self, name: &str) -> Option<HookTensor> {
        self.tensors.get(name).cloned()
    }
}

fn tensor_view_to_vec(view: &TensorView<'_>) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|chunk| {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(bytes)
        })
        .collect()
}

fn tensor_from_data_1d<B: Backend>(
    tensor: &HookTensor,
    device: &B::Device,
) -> Result<Tensor<B, 1>, Box<dyn std::error::Error>> {
    let shape: [usize; 1] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|_| "unexpected input rank")?;
    let data = Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device);
    Ok(data.reshape([shape[0] as i32]))
}

fn tensor_from_data_3d<B: Backend>(
    tensor: &HookTensor,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let shape: [usize; 3] = tensor
        .shape
        .clone()
        .try_into()
        .map_err(|_| "unexpected input rank")?;
    let data = Tensor::<B, 1>::from_floats(tensor.data.as_slice(), device);
    Ok(data.reshape([shape[0] as i32, shape[1] as i32, shape[2] as i32]))
}

struct MetricStats {
    mean_abs: f32,
    max_abs: f32,
    mse: f32,
}

fn compute_stats(burn: &[f32], reference: &[f32]) -> MetricStats {
    let mut sum_abs = 0.0f32;
    let mut max_abs = 0.0f32;
    let mut mse = 0.0f32;

    for (&lhs, &rhs) in burn.iter().zip(reference.iter()) {
        let diff = lhs - rhs;
        let abs = diff.abs();
        sum_abs += abs;
        max_abs = max_abs.max(abs);
        mse += diff * diff;
    }

    let len = burn.len().max(1) as f32;
    MetricStats {
        mean_abs: sum_abs / len,
        max_abs,
        mse: mse / len,
    }
}

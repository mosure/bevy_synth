use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::prelude::*;
use clap::Parser;
use safetensors::tensor::{SafeTensors, TensorView};
use serde::{Deserialize, Serialize};

use burn_tripo::model::triposg::{
    hooks::HookRecorder,
    vae::{TripoSGVaeConfig, import::load_triposg_vae},
};

#[derive(Clone, Debug, Serialize, Deserialize, Parser)]
#[command(about = "Export TripoSG VAE hooks to safetensors", version, long_about = None)]
struct HookConfig {
    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long, default_value = "assets/hooks/triposg_vae_input.safetensors")]
    inputs: PathBuf,

    #[arg(long, default_value = "assets/hooks/triposg_vae_burn.safetensors")]
    output: PathBuf,

    #[arg(long)]
    use_mean: bool,
}

type BackendImpl = burn::backend::NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = HookConfig::parse();
    let device = <BackendImpl as burn::tensor::backend::BackendTypes>::Device::default();
    let inputs = resolve_io_path(args.inputs);
    let output = resolve_output_path(args.output);
    let weights = resolve_weights_path(args.weights, "vae/diffusion_pytorch_model.safetensors");

    let bytes = fs::read(&inputs)?;
    let tensors = SafeTensors::deserialize(&bytes)?;

    let coords = tensor_from_view::<BackendImpl>(&tensors, "input.coords", &device)?;
    let features = tensor_from_view::<BackendImpl>(&tensors, "input.features", &device)?;
    let query_coords = tensor_from_view::<BackendImpl>(&tensors, "input.query_coords", &device)?;

    let config = weights
        .parent()
        .and_then(|path| TripoSGVaeConfig::from_config_file(path.join("config.json")).ok())
        .unwrap_or_else(TripoSGVaeConfig::midi_3d);
    let model = load_triposg_vae::<BackendImpl>(&config, &device, &weights)?;

    let mut hooks = HookRecorder::new();
    hooks.record_tensor("input.coords", &coords);
    hooks.record_tensor("input.features", &features);
    hooks.record_tensor("input.query_coords", &query_coords);

    let _output = model.forward(
        coords,
        features,
        query_coords,
        args.use_mean,
        Some(&mut hooks),
    );
    hooks.write_safetensors(&output)?;

    println!("Saved hook outputs to {}", output.display());
    Ok(())
}

fn resolve_weights_path(arg: Option<PathBuf>, leaf: &str) -> PathBuf {
    if let Some(path) = arg {
        return path;
    }
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let candidate = PathBuf::from(root).join(leaf);
        if candidate.exists() {
            return candidate;
        }
    }
    let tripo_root = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG");
    let candidate = tripo_root.join(leaf);
    if candidate.exists() {
        return candidate;
    }
    manifest_path("assets/models/MIDI-3D").join(leaf)
}

fn manifest_path(path: impl AsRef<Path>) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn resolve_io_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() || path.exists() {
        return path;
    }
    let candidate = manifest_path(&path);
    if candidate.exists() {
        return candidate;
    }
    path
}

fn resolve_output_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    manifest_path(path)
}

fn tensor_from_view<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 3>, Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|_| format!("missing tensor `{name}` in input safetensors"))?;
    let shape: [usize; 3] = view
        .shape()
        .try_into()
        .map_err(|_| format!("unexpected rank for `{name}`"))?;
    let data = tensor_view_to_vec(&view);
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([shape[0] as i32, shape[1] as i32, shape[2] as i32]))
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

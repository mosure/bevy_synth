use std::{
    fs,
    path::{Path, PathBuf},
};

use bevy_args::{Deserialize, Parser, Serialize, parse_args};
use burn::prelude::*;
use safetensors::tensor::{SafeTensors, TensorView};

use burn_3d_synth_tripo::model::triposg::{
    hooks::HookRecorder,
    image_encoder::import::{load_dinov2_processor, load_triposg_dinov2},
};

#[derive(Clone, Debug, Serialize, Deserialize, Parser)]
#[command(about = "Export TripoSG DINOv2 hooks to safetensors", version, long_about = None)]
struct HookConfig {
    #[arg(long)]
    weights: Option<PathBuf>,

    #[arg(long, default_value = "assets/hooks/triposg_dino_input.safetensors")]
    inputs: PathBuf,

    #[arg(long, default_value = "assets/hooks/triposg_dino_burn.safetensors")]
    output: PathBuf,
}

type BackendImpl = burn::backend::NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args::<HookConfig>();
    let device = <BackendImpl as burn::tensor::backend::Backend>::Device::default();

    let inputs = resolve_io_path(args.inputs);
    let output_path = resolve_output_path(args.output);
    let weights_path = resolve_weights_path(args.weights.as_ref());
    let weights_root = resolve_weights_root_from_path(&weights_path);

    let bytes = fs::read(&inputs)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let image = tensor_from_view_4d::<BackendImpl>(&tensors, "input.image", &device)?;

    let image_encoder = load_triposg_dinov2::<BackendImpl>(&device, &weights_path)?;
    let image_processor = load_dinov2_processor(weights_root)?;

    let processed = image_processor.preprocess(image.clone());
    let dino_output = image_encoder.dino.forward(processed.clone(), None);

    let cls = dino_output.x_norm_clstoken.unsqueeze_dim(1);
    let patch = dino_output.x_norm_patchtokens;
    let image_embeds = Tensor::cat(vec![cls.clone(), patch.clone()], 1);

    let mut hooks = HookRecorder::new();
    hooks.record_tensor("input.image", &image);
    hooks.record_tensor("image.preprocessed", &processed);
    hooks.record_tensor("output.image_embeds", &image_embeds);
    hooks.record_tensor("output.cls_token", &cls);
    hooks.record_tensor("output.patch_tokens", &patch);
    hooks.write_safetensors(&output_path)?;

    println!("Saved hook outputs to {}", output_path.display());
    Ok(())
}

fn resolve_weights_path(arg: Option<&PathBuf>) -> PathBuf {
    if let Some(path) = arg {
        if path.is_dir() {
            let candidate = path.join("model.safetensors");
            if candidate.exists() {
                return candidate;
            }
            let nested = path.join("image_encoder_dinov2").join("model.safetensors");
            if nested.exists() {
                return nested;
            }
        }
        return path.clone();
    }
    if let Ok(root) = std::env::var("TRIPOSG_WEIGHTS_ROOT") {
        let candidate = Path::new(&root)
            .join("image_encoder_dinov2")
            .join("model.safetensors");
        if candidate.exists() {
            return candidate;
        }
    }
    let tripo_root = PathBuf::from(r"E:\repos\TripoSG\pretrained_weights\TripoSG");
    let candidate = tripo_root
        .join("image_encoder_dinov2")
        .join("model.safetensors");
    if candidate.exists() {
        return candidate;
    }
    manifest_path("assets/models/MIDI-3D")
        .join("image_encoder_dinov2")
        .join("model.safetensors")
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

fn resolve_weights_root_from_path(weights_path: &Path) -> PathBuf {
    if weights_path.is_dir() {
        if weights_path.join("feature_extractor_dinov2").exists() {
            return weights_path.to_path_buf();
        }
        if weights_path
            .file_name()
            .map(|name| name == "image_encoder_dinov2")
            .unwrap_or(false)
            && let Some(parent) = weights_path.parent()
        {
            return parent.to_path_buf();
        }
        return weights_path.to_path_buf();
    }

    let parent = weights_path.parent().unwrap_or_else(|| Path::new("."));
    if parent
        .file_name()
        .map(|name| name == "image_encoder_dinov2")
        .unwrap_or(false)
        && let Some(root) = parent.parent()
    {
        return root.to_path_buf();
    }
    parent.to_path_buf()
}

fn tensor_from_view_4d<B: Backend>(
    tensors: &SafeTensors<'_>,
    name: &str,
    device: &B::Device,
) -> Result<Tensor<B, 4>, Box<dyn std::error::Error>> {
    let view = tensors
        .tensor(name)
        .map_err(|_| format!("missing tensor `{name}` in input safetensors"))?;
    let shape: [usize; 4] = view
        .shape()
        .try_into()
        .map_err(|_| format!("unexpected rank for `{name}`"))?;
    let data = tensor_view_to_vec(&view);
    let flat = Tensor::<B, 1>::from_floats(data.as_slice(), device);
    Ok(flat.reshape([
        shape[0] as i32,
        shape[1] as i32,
        shape[2] as i32,
        shape[3] as i32,
    ]))
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

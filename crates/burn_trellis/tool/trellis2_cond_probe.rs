use std::fs;
use std::path::PathBuf;

use burn_trellis::hook_diff::HookSnapshot;
use burn_trellis::preprocess::PreprocessOutput;
use burn_trellis::runtime_model::image_conditioning::extract_condition_from_model_name;
use clap::Parser;
use safetensors::{Dtype, serialize, tensor::TensorView};

#[derive(Parser, Debug)]
#[command(about = "Run burn_trellis image-conditioning runtime on a captured preprocess tensor.")]
struct Args {
    #[arg(long)]
    preprocess_hook: PathBuf,
    #[arg(long)]
    output_hook: PathBuf,
    #[arg(long)]
    weights_root: PathBuf,
    #[arg(long)]
    image_large_root: Option<PathBuf>,
    #[arg(long, default_value = "facebook/dinov3-vitl16-pretrain-lvd1689m")]
    model_name: String,
    #[arg(long, default_value_t = 512)]
    resolution: usize,
    #[arg(long, default_value_t = true)]
    prefer_wgpu: bool,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    if args.resolution == 0 {
        return Err("resolution must be > 0".to_string());
    }

    let snapshot = HookSnapshot::from_file(&args.preprocess_hook).map_err(|err| {
        format!(
            "failed to load preprocess hook '{}': {err}",
            args.preprocess_hook.display()
        )
    })?;
    let preprocess = preprocess_from_snapshot(&snapshot)?;

    let (output, backend_name) = extract_condition_from_model_name(
        &args.weights_root,
        args.image_large_root.as_deref(),
        args.model_name.as_str(),
        args.prefer_wgpu,
        &preprocess,
        args.resolution,
    )?;

    let key_root = match args.resolution {
        512 => "get_cond_512".to_string(),
        1024 => "get_cond_1024".to_string(),
        other => format!("get_cond_{other}"),
    };
    let shape = vec![1usize, output.token_count, output.channels];
    let cond_bytes = f32_values_to_bytes(output.values.as_slice());
    let neg_bytes = vec![0u8; cond_bytes.len()];

    let cond_view = TensorView::new(Dtype::F32, shape.clone(), cond_bytes.as_slice())
        .map_err(|err| format!("failed to build cond tensor view: {err}"))?;
    let neg_view = TensorView::new(Dtype::F32, shape.clone(), neg_bytes.as_slice())
        .map_err(|err| format!("failed to build neg_cond tensor view: {err}"))?;

    let serialized = serialize(
        vec![
            (format!("{key_root}.out.cond"), cond_view),
            (format!("{key_root}.out.neg_cond"), neg_view),
        ],
        None,
    )
    .map_err(|err| format!("failed to serialize output hook: {err}"))?;
    if let Some(parent) = args.output_hook.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create output directory '{}': {err}",
                parent.display()
            )
        })?;
    }
    fs::write(&args.output_hook, serialized).map_err(|err| {
        format!(
            "failed to write output hook '{}': {err}",
            args.output_hook.display()
        )
    })?;

    println!(
        "trellis2_cond_probe: backend={} resolution={} token_count={} channels={} output={}",
        backend_name,
        output.resolution,
        output.token_count,
        output.channels,
        args.output_hook.display()
    );
    Ok(())
}

fn preprocess_from_snapshot(snapshot: &HookSnapshot) -> Result<PreprocessOutput, String> {
    let tensor = snapshot
        .tensors
        .get("preprocess_image.output")
        .ok_or_else(|| "missing preprocess_image.output tensor in hook".to_string())?;
    if tensor.shape.len() != 3 || tensor.shape[2] != 3 {
        return Err(format!(
            "invalid preprocess_image.output shape {:?}; expected [H, W, 3]",
            tensor.shape
        ));
    }
    let height = tensor.shape[0];
    let width = tensor.shape[1];
    let expected = height.saturating_mul(width).saturating_mul(3);
    if tensor.data.len() != expected {
        return Err(format!(
            "preprocess_image.output element count mismatch: got {}, expected {}",
            tensor.data.len(),
            expected
        ));
    }
    let rgb = tensor
        .data
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect::<Vec<_>>();
    Ok(PreprocessOutput {
        width: width as u32,
        height: height as u32,
        rgb,
    })
}

fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(value.to_le_bytes().as_slice());
    }
    bytes
}

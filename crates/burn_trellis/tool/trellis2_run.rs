use std::path::PathBuf;

use burn_trellis::TrellisQuality;
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use clap::Parser;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(about = "Run burn_trellis pipeline and optionally emit OBJ + safetensors hook")]
struct Args {
    /// Input image path.
    #[arg(long)]
    input_image: PathBuf,

    /// Optional OBJ output path.
    #[arg(long)]
    output_obj: Option<PathBuf>,

    /// Optional safetensors hook output path.
    #[arg(long)]
    hook_output: Option<PathBuf>,

    /// Optional safetensors hook input path used as deterministic stage-noise overrides.
    #[arg(long)]
    noise_overrides_hook: Option<PathBuf>,

    /// Optional Trellis2 weights root (defaults to env/probed root).
    #[arg(long)]
    weights_root: Option<PathBuf>,

    /// Optional TRELLIS-image-large root.
    #[arg(long)]
    image_large_root: Option<PathBuf>,

    /// Runtime quality preset.
    #[arg(long, value_enum, default_value_t = TrellisQuality::Medium)]
    quality: TrellisQuality,

    /// Runtime device target.
    #[arg(long, value_enum, default_value_t = TrellisDevice::Auto)]
    device: TrellisDevice,

    /// Optional deterministic seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Fail if sparse-structure stage falls back to synthetic mode.
    #[arg(long, default_value_t = false)]
    require_runtime_model: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut config = Trellis2PipelineConfig::default();
    if let Some(path) = args.weights_root {
        config.weights_root = path;
    }
    if let Some(path) = args.image_large_root {
        config.image_large_root = Some(path);
    }

    let pipeline = Trellis2Pipeline::new(config)?;
    pipeline.validate_runtime()?;

    let options = TrellisRunOptions {
        quality: args.quality,
        device: args.device,
        seed: args.seed,
        hook_output: args.hook_output.clone(),
        noise_overrides_hook: args.noise_overrides_hook.clone(),
    };

    let profiled = pipeline.infer_mesh_profile(&args.input_image, &options)?;
    if args.require_runtime_model && profiled.sparse_source.as_str() == "synthetic" {
        return Err("runtime-model required but sparse stage used synthetic fallback".into());
    }
    if let Some(obj_path) = args.output_obj.as_ref() {
        burn_trellis::write_obj_mesh(obj_path, &profiled.mesh)?;
    }

    println!(
        "{}",
        json!({
            "status": "ok",
            "vertices": profiled.mesh.vertices.len(),
            "faces": profiled.mesh.faces.len(),
            "elapsed_ms": profiled.timings.total_ms,
            "device": options.device.as_str(),
            "quality": options.quality.as_str(),
            "sparse_source": profiled.sparse_source.as_str(),
            "timings_ms": {
                "preprocess": profiled.timings.preprocess_ms,
                "runtime_setup": profiled.timings.runtime_setup_ms,
                "sparse": profiled.timings.sparse_ms,
                "shape_slat": profiled.timings.shape_slat_ms,
                "tex_slat": profiled.timings.tex_slat_ms,
                "decode": profiled.timings.decode_ms,
                "hook_capture": profiled.timings.hook_capture_ms,
                "host_readback_count": profiled.timings.host_readback_count,
                "host_readback_elements": profiled.timings.host_readback_elements,
                "total": profiled.timings.total_ms
            },
            "hook_output": args.hook_output,
            "noise_overrides_hook": args.noise_overrides_hook,
            "output_obj": args.output_obj,
        })
    );
    Ok(())
}

use std::path::{Path, PathBuf};

use burn_trellis::TrellisQuality;
use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use clap::Parser;
use serde_json::json;

#[derive(Parser, Debug)]
#[command(about = "Run burn_trellis pipeline and optionally emit OBJ/GLB + safetensors hook")]
struct Args {
    /// Input image path.
    #[arg(long = "input", alias = "input-image")]
    input: PathBuf,

    /// Optional GLB output path. If omitted, no GLB is written.
    /// If the path has no extension (or is a directory), `<input>_mesh.glb` is used.
    #[arg(long = "output", alias = "output-glb")]
    output: Option<PathBuf>,

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
    #[arg(long = "weights-root", alias = "trellis-weights-root")]
    weights_root: Option<PathBuf>,

    /// Optional TRELLIS-image-large root.
    #[arg(long = "image-large-root", alias = "trellis-image-large-root")]
    image_large_root: Option<PathBuf>,

    /// Runtime quality preset.
    #[arg(long, value_enum, default_value_t = TrellisQuality::Medium)]
    quality: TrellisQuality,

    /// Runtime backend target.
    #[arg(long = "backend", alias = "device", value_enum, default_value_t = TrellisDevice::Auto)]
    backend: TrellisDevice,

    /// Optional deterministic seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Fail if sparse-structure stage falls back to synthetic mode.
    #[arg(long, default_value_t = false)]
    require_runtime_model: bool,

    /// Fail if any pipeline stage uses fallback behavior (benchmark strict mode).
    #[arg(long, default_value_t = false)]
    strict_benchmark: bool,

    /// Number of repeated in-process runs (reuses the loaded runtime cache).
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Optional cap on sparse coords before decode. Use lower values for short strict passes.
    #[arg(long)]
    max_sparse_coords: Option<usize>,
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let output_glb = args
        .output
        .as_ref()
        .map(|path| resolve_glb_output_path(path.as_path(), args.input.as_path()));

    let mut config = Trellis2PipelineConfig::default();
    if let Some(path) = args.weights_root.as_ref() {
        config.weights_root = path.clone();
    }
    if let Some(path) = args.image_large_root.as_ref() {
        config.image_large_root = Some(path.clone());
    }

    let pipeline = Trellis2Pipeline::new(config).map_err(|err| err.to_string())?;
    pipeline.validate_runtime().map_err(|err| err.to_string())?;

    let repeat = args.repeat.max(1);
    let mut runs = Vec::with_capacity(repeat);
    for run_idx in 0..repeat {
        let options = TrellisRunOptions {
            quality: args.quality,
            device: args.backend,
            seed: args.seed,
            hook_output: if run_idx + 1 == repeat {
                args.hook_output.clone()
            } else {
                None
            },
            noise_overrides_hook: args.noise_overrides_hook.clone(),
            max_sparse_coords: args.max_sparse_coords,
        };
        let profiled = pipeline
            .infer_mesh_profile(&args.input, &options)
            .map_err(|err| err.to_string())?;
        if args.require_runtime_model && profiled.sparse_source.as_str() == "synthetic" {
            return Err(
                "runtime-model required but sparse stage used synthetic fallback".to_string(),
            );
        }
        if args.strict_benchmark {
            if profiled.sparse_source.as_str() == "synthetic" {
                return Err(
                    "strict benchmark failed: sparse stage used synthetic fallback".to_string(),
                );
            }
            if matches!(args.backend, TrellisDevice::Wgpu)
                && profiled.sparse_source.as_str() != "runtime_model_wgpu"
            {
                return Err(format!(
                    "strict benchmark failed: requested wgpu but sparse stage source was '{}'",
                    profiled.sparse_source.as_str()
                ));
            }
            if profiled.decode_source.is_fallback() {
                return Err(format!(
                    "strict benchmark failed: decode stage used fallback source '{}'",
                    profiled.decode_source.as_str()
                ));
            }
            if matches!(args.backend, TrellisDevice::Wgpu)
                && profiled.timings.decode_shape_wgpu_dispatches == 0
                && profiled.timings.decode_tex_wgpu_dispatches == 0
                && profiled.decode_source.as_str() != "runtime_hook_override"
            {
                return Err(
                    "strict benchmark failed: requested wgpu but decode emitted zero wgpu dispatches"
                        .to_string(),
                );
            }
        }
        runs.push(profiled);
    }
    let profiled = runs
        .last()
        .ok_or_else(|| "no inference runs were produced".to_string())?;
    if let Some(obj_path) = args.output_obj.as_ref() {
        burn_trellis::write_obj_mesh(obj_path, &profiled.mesh).map_err(|err| err.to_string())?;
    }
    if let Some(glb_path) = output_glb.as_ref() {
        burn_trellis::write_glb_mesh(glb_path, &profiled.mesh).map_err(|err| err.to_string())?;
    }

    let timings_json = |profiled: &burn_trellis::pipeline::TrellisInferenceProfile| {
        json!({
            "preprocess": profiled.timings.preprocess_ms,
            "runtime_setup": profiled.timings.runtime_setup_ms,
            "sparse": profiled.timings.sparse_ms,
            "sparse_cond": profiled.timings.sparse_cond_ms,
            "sparse_sample": profiled.timings.sparse_sample_ms,
            "sparse_post": profiled.timings.sparse_post_ms,
            "shape_slat": profiled.timings.shape_slat_ms,
            "tex_slat": profiled.timings.tex_slat_ms,
            "decode": profiled.timings.decode_ms,
            "decode_shape_decoder": profiled.timings.decode_shape_decoder_ms,
            "decode_tex_decoder": profiled.timings.decode_tex_decoder_ms,
            "decode_attr_merge": profiled.timings.decode_attr_merge_ms,
            "decode_mesh": profiled.timings.decode_mesh_ms,
            "decode_pbr": profiled.timings.decode_pbr_ms,
            "decode_shape_conv_calls": profiled.timings.decode_shape_conv_calls,
            "decode_tex_conv_calls": profiled.timings.decode_tex_conv_calls,
            "decode_shape_wgpu_dispatches": profiled.timings.decode_shape_wgpu_dispatches,
            "decode_tex_wgpu_dispatches": profiled.timings.decode_tex_wgpu_dispatches,
            "decode_shape_wgpu_chunked_calls": profiled.timings.decode_shape_wgpu_chunked_calls,
            "decode_tex_wgpu_chunked_calls": profiled.timings.decode_tex_wgpu_chunked_calls,
            "decode_shape_wgpu_input_bytes": profiled.timings.decode_shape_wgpu_input_bytes,
            "decode_tex_wgpu_input_bytes": profiled.timings.decode_tex_wgpu_input_bytes,
            "decode_shape_wgpu_output_bytes": profiled.timings.decode_shape_wgpu_output_bytes,
            "decode_tex_wgpu_output_bytes": profiled.timings.decode_tex_wgpu_output_bytes,
            "decode_shape_wgpu_max_chunk_rows": profiled.timings.decode_shape_wgpu_max_chunk_rows,
            "decode_tex_wgpu_max_chunk_rows": profiled.timings.decode_tex_wgpu_max_chunk_rows,
            "hook_capture": profiled.timings.hook_capture_ms,
            "host_readback_count": profiled.timings.host_readback_count,
            "host_readback_elements": profiled.timings.host_readback_elements,
            "total": profiled.timings.total_ms
        })
    };

    if repeat == 1 {
        println!(
            "{}",
            json!({
                "status": "ok",
                "vertices": profiled.mesh.vertices.len(),
                "faces": profiled.mesh.faces.len(),
                "elapsed_ms": profiled.timings.total_ms,
                "backend": args.backend.as_str(),
                "device": args.backend.as_str(),
                "quality": args.quality.as_str(),
                "strict_benchmark": args.strict_benchmark,
                "repeat": repeat,
                "max_sparse_coords": args.max_sparse_coords,
                "sparse_source": profiled.sparse_source.as_str(),
                "decode_source": profiled.decode_source.as_str(),
                "fallbacks": {
                    "sparse": profiled.sparse_source.as_str() == "synthetic",
                    "decode": profiled.decode_source.is_fallback(),
                },
                "timings_ms": timings_json(profiled),
                "hook_output": args.hook_output,
                "noise_overrides_hook": args.noise_overrides_hook,
                "output_obj": args.output_obj,
                "output_glb": output_glb,
            })
        );
    } else {
        let runs_json = runs
            .iter()
            .enumerate()
            .map(|(idx, run)| {
                json!({
                    "run": idx + 1,
                    "elapsed_ms": run.timings.total_ms,
                    "runtime_setup_ms": run.timings.runtime_setup_ms,
                    "vertices": run.mesh.vertices.len(),
                    "faces": run.mesh.faces.len(),
                    "sparse_source": run.sparse_source.as_str(),
                    "decode_source": run.decode_source.as_str(),
                    "timings_ms": timings_json(run),
                })
            })
            .collect::<Vec<_>>();
        let setup_mean = runs
            .iter()
            .map(|run| run.timings.runtime_setup_ms)
            .sum::<f64>()
            / runs.len() as f64;
        let total_mean =
            runs.iter().map(|run| run.timings.total_ms).sum::<f64>() / runs.len() as f64;
        println!(
            "{}",
            json!({
                "status": "ok",
                "backend": args.backend.as_str(),
                "device": args.backend.as_str(),
                "quality": args.quality.as_str(),
                "strict_benchmark": args.strict_benchmark,
                "repeat": repeat,
                "max_sparse_coords": args.max_sparse_coords,
                "summary": {
                    "runtime_setup_mean_ms": setup_mean,
                    "total_mean_ms": total_mean,
                },
                "last": {
                    "vertices": profiled.mesh.vertices.len(),
                    "faces": profiled.mesh.faces.len(),
                    "elapsed_ms": profiled.timings.total_ms,
                    "sparse_source": profiled.sparse_source.as_str(),
                    "decode_source": profiled.decode_source.as_str(),
                    "timings_ms": timings_json(profiled),
                },
                "runs": runs_json,
                "hook_output": args.hook_output,
                "noise_overrides_hook": args.noise_overrides_hook,
                "output_obj": args.output_obj,
                "output_glb": output_glb,
            })
        );
    }
    Ok(())
}

fn resolve_glb_output_path(output: &Path, input: &Path) -> PathBuf {
    if output.extension().is_none() || output.is_dir() {
        let stem = input
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("mesh");
        return output.join(format!("{stem}_mesh.glb"));
    }
    if output
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("glb"))
        .unwrap_or(false)
    {
        output.to_path_buf()
    } else {
        output.with_extension("glb")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = std::thread::Builder::new()
        .name("trellis2_run".to_string())
        .stack_size(64 * 1024 * 1024)
        .spawn(run)?;
    match handle.join() {
        Ok(result) => result.map_err(|err| err.into()),
        Err(payload) => {
            let message = if let Some(message) = payload.downcast_ref::<&str>() {
                (*message).to_string()
            } else if let Some(message) = payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "trellis2_run worker thread panicked with non-string payload".to_string()
            };
            Err(message.into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_burn_synth_style_aliases() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input",
            "input.png",
            "--output",
            "out_mesh",
            "--backend",
            "wgpu",
            "--weights-root",
            "weights",
            "--image-large-root",
            "image_large",
        ]);
        assert_eq!(args.input, PathBuf::from("input.png"));
        assert_eq!(args.output, Some(PathBuf::from("out_mesh")));
        assert!(matches!(args.backend, TrellisDevice::Wgpu));
        assert_eq!(args.weights_root, Some(PathBuf::from("weights")));
        assert_eq!(args.image_large_root, Some(PathBuf::from("image_large")));
    }

    #[test]
    fn accepts_legacy_trellis2_run_aliases() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input-image",
            "input.png",
            "--output-glb",
            "mesh_out",
            "--device",
            "cpu",
            "--trellis-weights-root",
            "weights",
            "--trellis-image-large-root",
            "image_large",
        ]);
        assert_eq!(args.input, PathBuf::from("input.png"));
        assert_eq!(args.output, Some(PathBuf::from("mesh_out")));
        assert!(matches!(args.backend, TrellisDevice::Cpu));
        assert_eq!(args.weights_root, Some(PathBuf::from("weights")));
        assert_eq!(args.image_large_root, Some(PathBuf::from("image_large")));
    }

    #[test]
    fn output_path_resolution_matches_burn_synth_behavior() {
        let input = Path::new("chair.png");
        assert_eq!(
            resolve_glb_output_path(Path::new("tmp/out"), input),
            PathBuf::from("tmp/out/chair_mesh.glb")
        );
        assert_eq!(
            resolve_glb_output_path(Path::new("tmp/out.obj"), input),
            PathBuf::from("tmp/out.glb")
        );
        assert_eq!(
            resolve_glb_output_path(Path::new("tmp/out.glb"), input),
            PathBuf::from("tmp/out.glb")
        );
    }
}

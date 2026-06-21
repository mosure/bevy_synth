#![recursion_limit = "256"]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use burn_trellis::pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions,
};
use burn_trellis::staged_pipeline::TrellisSamplerRuntimeOverrides;
use burn_trellis::{TrellisComputeProfile, TrellisQuality};
use clap::Parser;
use serde_json::json;

const OVOXEL_MIN_SOURCE_VERTICES: usize = 1_000_000;

fn ns_to_ms(ns: u64) -> f64 {
    ns as f64 / 1_000_000.0
}

fn flow_ops_json(
    ops: burn_trellis::staged_pipeline::SparseFlowOpTimingSummary,
) -> serde_json::Value {
    json!({
        "self_attn_calls": ops.self_attn_calls,
        "self_attn_ns": ops.self_attn_ns,
        "self_attn_ms": ns_to_ms(ops.self_attn_ns),
        "cross_attn_calls": ops.cross_attn_calls,
        "cross_attn_ns": ops.cross_attn_ns,
        "cross_attn_ms": ns_to_ms(ops.cross_attn_ns),
        "mlp_calls": ops.mlp_calls,
        "mlp_ns": ops.mlp_ns,
        "mlp_ms": ns_to_ms(ops.mlp_ns),
        "self_qkv_calls": ops.self_qkv_calls,
        "self_qkv_ns": ops.self_qkv_ns,
        "self_qkv_ms": ns_to_ms(ops.self_qkv_ns),
        "self_norm_rope_calls": ops.self_norm_rope_calls,
        "self_norm_rope_ns": ops.self_norm_rope_ns,
        "self_norm_rope_ms": ns_to_ms(ops.self_norm_rope_ns),
        "self_norm_rope_fused_qk_calls": ops.self_norm_rope_fused_qk_calls,
        "self_norm_rope_fused_qkv_module_calls": ops.self_norm_rope_fused_qkv_module_calls,
        "self_kernel_calls": ops.self_kernel_calls,
        "self_kernel_ns": ops.self_kernel_ns,
        "self_kernel_ms": ns_to_ms(ops.self_kernel_ns),
        "self_out_calls": ops.self_out_calls,
        "self_out_ns": ops.self_out_ns,
        "self_out_ms": ns_to_ms(ops.self_out_ns),
        "self_cat_calls": ops.self_cat_calls,
        "self_cat_ns": ops.self_cat_ns,
        "self_cat_ms": ns_to_ms(ops.self_cat_ns),
        "cross_q_calls": ops.cross_q_calls,
        "cross_q_ns": ops.cross_q_ns,
        "cross_q_ms": ns_to_ms(ops.cross_q_ns),
        "cross_kv_calls": ops.cross_kv_calls,
        "cross_kv_ns": ops.cross_kv_ns,
        "cross_kv_ms": ns_to_ms(ops.cross_kv_ns),
        "cross_norm_calls": ops.cross_norm_calls,
        "cross_norm_ns": ops.cross_norm_ns,
        "cross_norm_ms": ns_to_ms(ops.cross_norm_ns),
        "cross_kernel_calls": ops.cross_kernel_calls,
        "cross_kernel_ns": ops.cross_kernel_ns,
        "cross_kernel_ms": ns_to_ms(ops.cross_kernel_ns),
        "cross_out_calls": ops.cross_out_calls,
        "cross_out_ns": ops.cross_out_ns,
        "cross_out_ms": ns_to_ms(ops.cross_out_ns),
        "cross_cat_calls": ops.cross_cat_calls,
        "cross_cat_ns": ops.cross_cat_ns,
        "cross_cat_ms": ns_to_ms(ops.cross_cat_ns),
        "module_cast_pad_calls": ops.module_cast_pad_calls,
        "module_cast_pad_ns": ops.module_cast_pad_ns,
        "module_cast_pad_ms": ns_to_ms(ops.module_cast_pad_ns),
        "module_attention_calls": ops.module_attention_calls,
        "module_attention_ns": ops.module_attention_ns,
        "module_attention_ms": ns_to_ms(ops.module_attention_ns),
        "module_output_calls": ops.module_output_calls,
        "module_output_ns": ops.module_output_ns,
        "module_output_ms": ns_to_ms(ops.module_output_ns),
        "block_norm_mod_calls": ops.block_norm_mod_calls,
        "block_norm_mod_ns": ops.block_norm_mod_ns,
        "block_norm_mod_ms": ns_to_ms(ops.block_norm_mod_ns),
        "block_norm_affine_calls": ops.block_norm_affine_calls,
        "block_norm_affine_ns": ops.block_norm_affine_ns,
        "block_norm_affine_ms": ns_to_ms(ops.block_norm_affine_ns),
        "block_gate_residual_calls": ops.block_gate_residual_calls,
        "block_gate_residual_ns": ops.block_gate_residual_ns,
        "block_gate_residual_ms": ns_to_ms(ops.block_gate_residual_ns),
        "model_io_calls": ops.model_io_calls,
        "model_io_ns": ops.model_io_ns,
        "model_io_ms": ns_to_ms(ops.model_io_ns),
        "model_input_calls": ops.model_input_calls,
        "model_input_ns": ops.model_input_ns,
        "model_input_ms": ns_to_ms(ops.model_input_ns),
        "model_output_calls": ops.model_output_calls,
        "model_output_ns": ops.model_output_ns,
        "model_output_ms": ns_to_ms(ops.model_output_ns),
    })
}

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

    /// Optional JSON report output path for machine-readable timings.
    #[arg(long)]
    report_json: Option<PathBuf>,

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

    /// Runtime compute profile. Use reference-f32 for strict parity and wgpu-fast-f16 for parity-checked WGPU f16 attention.
    #[arg(long = "compute-profile", value_enum, default_value_t = TrellisComputeProfile::ReferenceF32)]
    compute_profile: TrellisComputeProfile,

    /// Runtime backend target.
    #[arg(long = "backend", alias = "device", value_enum, default_value_t = TrellisDevice::Auto)]
    backend: TrellisDevice,

    /// Optional deterministic seed.
    #[arg(long)]
    seed: Option<u64>,

    /// Fail unless sparse-structure stage runs through runtime model path.
    #[arg(long, default_value_t = false)]
    require_runtime_model: bool,

    /// Enforce strict runtime kernel-path invariants (benchmark strict mode).
    #[arg(long, default_value_t = false)]
    strict_benchmark: bool,

    /// Number of repeated in-process runs (reuses the loaded runtime cache).
    #[arg(long, default_value_t = 1)]
    repeat: usize,

    /// Optional cap on sparse coords before decode. Use lower values for short strict passes.
    #[arg(long)]
    max_sparse_coords: Option<usize>,

    /// Optional sparse-structure sampler step override for benchmark/profiling runs.
    #[arg(long)]
    sparse_steps: Option<usize>,

    /// Optional shape SLat sampler step override for benchmark/profiling runs.
    #[arg(long)]
    shape_steps: Option<usize>,

    /// Optional texture SLat sampler step override for benchmark/profiling runs.
    #[arg(long = "tex-steps", alias = "texture-steps")]
    tex_steps: Option<usize>,

    /// Enable runtime stage-level debug logs from canonical runtime-model path.
    #[arg(long, default_value_t = false)]
    runtime_stage_debug: bool,

    /// Enable runtime attention debug logs from canonical runtime-model path.
    #[arg(long, default_value_t = false)]
    runtime_attention_debug: bool,

    /// Emit decoder sparse-conv telemetry (variant/split/dispatch counters).
    #[arg(long, default_value_t = false)]
    runtime_decoder_conv_telemetry: bool,

    /// Export GLB via canonical o_voxel postprocess using hook tensors.
    #[arg(long, default_value_t = false)]
    ovoxel_postprocess_from_hook: bool,

    /// Python executable used for o_voxel postprocess export.
    #[arg(long, default_value = "python3")]
    ovoxel_python_bin: String,

    /// Optional override path for the o_voxel postprocess helper script.
    #[arg(long)]
    ovoxel_postprocess_script: Option<PathBuf>,

    /// o_voxel decimation target passed to postprocess export.
    #[arg(long, default_value_t = 1_000_000)]
    ovoxel_decimation_target: usize,

    /// o_voxel texture size passed to postprocess export.
    #[arg(long, default_value_t = 4096)]
    ovoxel_texture_size: usize,

    /// o_voxel remesh band passed to postprocess export.
    #[arg(long, default_value_t = 1.0)]
    ovoxel_remesh_band: f32,

    /// o_voxel remesh projection passed to postprocess export.
    #[arg(long, default_value_t = 0.0)]
    ovoxel_remesh_project: f32,

    /// Export GLB with WEBP extension textures (disabled by default for broader metric tooling compatibility).
    #[arg(long, default_value_t = false)]
    ovoxel_extension_webp: bool,
}

fn validate_benchmark_args(args: &Args) -> Result<(), String> {
    if args.strict_benchmark && args.runtime_stage_debug {
        return Err(
            "--strict-benchmark cannot be combined with --runtime-stage-debug; stage debug enables probe readbacks and invalidates timing"
                .to_string(),
        );
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    validate_benchmark_args(&args)?;
    let output_glb = args
        .output
        .as_ref()
        .map(|path| resolve_glb_output_path(path.as_path(), args.input.as_path()));
    if args.ovoxel_postprocess_from_hook && output_glb.is_none() {
        return Err(
            "ovoxel postprocess export requires --output/--output-glb to be provided".to_string(),
        );
    }
    if args.ovoxel_postprocess_from_hook && args.hook_output.is_none() {
        return Err("ovoxel postprocess export requires --hook-output".to_string());
    }

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
            compute_profile: args.compute_profile,
            seed: args.seed,
            hook_output: if run_idx + 1 == repeat {
                args.hook_output.clone()
            } else {
                None
            },
            noise_overrides_hook: args.noise_overrides_hook.clone(),
            max_sparse_coords: args.max_sparse_coords,
            target_faces: None,
            runtime_stage_debug: args.runtime_stage_debug,
            runtime_attention_debug: args.runtime_attention_debug,
            runtime_decoder_conv_telemetry: args.runtime_decoder_conv_telemetry,
            runtime_stage_fence: args.strict_benchmark,
            sampler_overrides: TrellisSamplerRuntimeOverrides {
                sparse_steps: args.sparse_steps,
                shape_steps: args.shape_steps,
                tex_steps: args.tex_steps,
                sparse_guidance_strength: None,
                shape_guidance_strength: None,
                tex_guidance_strength: None,
            },
        };
        let profiled = pipeline
            .infer_mesh_profile(&args.input, &options)
            .map_err(|err| err.to_string())?;
        let sparse_source = profiled.sparse_source.as_str();
        let decode_source = profiled.decode_source.as_str();
        let sparse_is_runtime_model =
            matches!(sparse_source, "runtime_model_cpu" | "runtime_model_wgpu");
        if args.require_runtime_model && !sparse_is_runtime_model {
            return Err(format!(
                "runtime-model required but sparse stage source was '{}'",
                sparse_source
            ));
        }
        if args.strict_benchmark {
            if !sparse_is_runtime_model {
                return Err(format!(
                    "strict benchmark failed: sparse stage source was '{}'",
                    sparse_source
                ));
            }
            if decode_source != "runtime" {
                return Err(format!(
                    "strict benchmark failed: decode stage source was '{}' (expected 'runtime')",
                    decode_source
                ));
            }
            if matches!(args.backend, TrellisDevice::Wgpu) && sparse_source != "runtime_model_wgpu"
            {
                return Err(format!(
                    "strict benchmark failed: requested wgpu but sparse stage source was '{}'",
                    sparse_source
                ));
            }
            if matches!(args.backend, TrellisDevice::Wgpu)
                && (profiled.timings.decode_shape_wgpu_dispatches == 0
                    || profiled.timings.decode_tex_wgpu_dispatches == 0)
            {
                return Err(
                    "strict benchmark failed: requested wgpu but runtime decode emitted zero dispatches for shape and/or tex decoder"
                        .to_string(),
                );
            }
        }
        runs.push(profiled);
    }
    let profiled = runs
        .last()
        .ok_or_else(|| "no inference runs were produced".to_string())?;
    let mut ovoxel_postprocess_applied = false;
    let mut ovoxel_postprocess_skipped_reason: Option<String> = None;
    if let Some(obj_path) = args.output_obj.as_ref() {
        burn_trellis::write_obj_mesh(obj_path, &profiled.mesh).map_err(|err| err.to_string())?;
    }
    if let Some(glb_path) = output_glb.as_ref() {
        if args.ovoxel_postprocess_from_hook {
            if let Some(reason) = ovoxel_skip_reason(profiled.mesh.vertices.len()) {
                eprintln!("burn_trellis: {reason}; writing runtime GLB output instead.");
                burn_trellis::write_glb_mesh(glb_path, &profiled.mesh)
                    .map_err(|err| err.to_string())?;
                ovoxel_postprocess_skipped_reason = Some(reason);
            } else {
                let hook_path = args
                    .hook_output
                    .as_ref()
                    .ok_or_else(|| "ovoxel postprocess requires --hook-output".to_string())?;
                run_ovoxel_postprocess(&args, hook_path.as_path(), glb_path.as_path())?;
                ovoxel_postprocess_applied = true;
            }
        } else {
            burn_trellis::write_glb_mesh(glb_path, &profiled.mesh)
                .map_err(|err| err.to_string())?;
        }
    }

    let timings_json = |profiled: &burn_trellis::pipeline::TrellisInferenceProfile| {
        json!({
            "preprocess": profiled.timings.preprocess_ms,
            "runtime_setup": profiled.timings.runtime_setup_ms,
            "sparse": profiled.timings.sparse_ms,
            "sparse_cond": profiled.timings.sparse_cond_ms,
            "sparse_sample": profiled.timings.sparse_sample_ms,
            "sparse_post": profiled.timings.sparse_post_ms,
            "sparse_flow_ops": flow_ops_json(profiled.timings.sparse_flow_ops),
            "shape_slat": profiled.timings.shape_slat_ms,
            "shape_slat_flow_ops": flow_ops_json(profiled.timings.shape_slat_flow_ops),
            "tex_slat": profiled.timings.tex_slat_ms,
            "tex_slat_flow_ops": flow_ops_json(profiled.timings.tex_slat_flow_ops),
            "decode": profiled.timings.decode_ms,
            "decode_stage_fenced": profiled.timings.decode_stage_fenced,
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
    let shapes_json = |profiled: &burn_trellis::pipeline::TrellisInferenceProfile| {
        json!({
            "sparse_coords": profiled.shapes.sparse_coords,
            "shape_slat_lr_rows": profiled.shapes.shape_slat_lr_rows,
            "shape_slat_rows": profiled.shapes.shape_slat_rows,
            "tex_slat_rows": profiled.shapes.tex_slat_rows,
            "cond_512_tokens": profiled.shapes.cond_512_tokens,
            "cond_1024_tokens": profiled.shapes.cond_1024_tokens,
        })
    };
    let settings = args.quality.settings();
    let effective_steps = |configured: Option<usize>, preset: usize| configured.unwrap_or(preset);

    let report = if repeat == 1 {
        json!({
            "status": "ok",
            "vertices": profiled.mesh.vertices.len(),
            "faces": profiled.mesh.faces.len(),
            "elapsed_ms": profiled.timings.total_ms,
            "backend": args.backend.as_str(),
            "device": args.backend.as_str(),
            "quality": args.quality.as_str(),
            "compute_profile": args.compute_profile.as_str(),
            "strict_benchmark": args.strict_benchmark,
            "repeat": repeat,
            "effective_config": {
                "pipeline_type": settings.pipeline_type,
                "max_num_tokens": settings.max_num_tokens,
                "sparse_steps": effective_steps(args.sparse_steps, settings.sparse_steps),
                "shape_steps": effective_steps(args.shape_steps, settings.shape_steps),
                "tex_steps": effective_steps(args.tex_steps, settings.texture_steps),
                "sparse_guidance": settings.guidance_strength_sparse,
                "shape_guidance": settings.guidance_strength_shape,
                "tex_guidance": settings.guidance_strength_texture,
            },
            "max_sparse_coords": args.max_sparse_coords,
            "sampler_overrides": {
                "sparse_steps": args.sparse_steps,
                "shape_steps": args.shape_steps,
                "tex_steps": args.tex_steps,
            },
            "ovoxel_postprocess_from_hook": args.ovoxel_postprocess_from_hook,
            "ovoxel_postprocess_applied": ovoxel_postprocess_applied,
            "ovoxel_postprocess_skipped_reason": ovoxel_postprocess_skipped_reason,
            "sparse_source": profiled.sparse_source.as_str(),
            "decode_source": profiled.decode_source.as_str(),
            "kernel_invariants": {
                "sparse_runtime_model": matches!(profiled.sparse_source.as_str(), "runtime_model_cpu" | "runtime_model_wgpu"),
                "decode_runtime": profiled.decode_source.as_str() == "runtime",
                "wgpu_shape_dispatches": profiled.timings.decode_shape_wgpu_dispatches,
                "wgpu_tex_dispatches": profiled.timings.decode_tex_wgpu_dispatches,
            },
            "shapes": shapes_json(profiled),
            "timings_ms": timings_json(profiled),
            "hook_output": args.hook_output,
            "noise_overrides_hook": args.noise_overrides_hook,
            "output_obj": args.output_obj,
            "output_glb": output_glb,
        })
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
                    "shapes": shapes_json(run),
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
        json!({
            "status": "ok",
            "backend": args.backend.as_str(),
            "device": args.backend.as_str(),
            "quality": args.quality.as_str(),
            "compute_profile": args.compute_profile.as_str(),
            "strict_benchmark": args.strict_benchmark,
            "repeat": repeat,
            "effective_config": {
                "pipeline_type": settings.pipeline_type,
                "max_num_tokens": settings.max_num_tokens,
                "sparse_steps": effective_steps(args.sparse_steps, settings.sparse_steps),
                "shape_steps": effective_steps(args.shape_steps, settings.shape_steps),
                "tex_steps": effective_steps(args.tex_steps, settings.texture_steps),
                "sparse_guidance": settings.guidance_strength_sparse,
                "shape_guidance": settings.guidance_strength_shape,
                "tex_guidance": settings.guidance_strength_texture,
            },
            "max_sparse_coords": args.max_sparse_coords,
            "sampler_overrides": {
                "sparse_steps": args.sparse_steps,
                "shape_steps": args.shape_steps,
                "tex_steps": args.tex_steps,
            },
            "ovoxel_postprocess_from_hook": args.ovoxel_postprocess_from_hook,
            "ovoxel_postprocess_applied": ovoxel_postprocess_applied,
            "ovoxel_postprocess_skipped_reason": ovoxel_postprocess_skipped_reason,
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
                "shapes": shapes_json(profiled),
                "timings_ms": timings_json(profiled),
            },
            "runs": runs_json,
            "hook_output": args.hook_output,
            "noise_overrides_hook": args.noise_overrides_hook,
            "output_obj": args.output_obj,
            "output_glb": output_glb,
        })
    };
    if let Some(path) = args.report_json.as_ref() {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|err| {
                format!("failed to create report parent {}: {err}", parent.display())
            })?;
        }
        fs::write(
            path,
            serde_json::to_vec_pretty(&report)
                .map_err(|err| format!("failed to serialize JSON report: {err}"))?,
        )
        .map_err(|err| format!("failed to write JSON report {}: {err}", path.display()))?;
    }
    println!("{report}");
    Ok(())
}

fn ovoxel_skip_reason(source_vertices: usize) -> Option<String> {
    if source_vertices < OVOXEL_MIN_SOURCE_VERTICES {
        return Some(format!(
            "o_voxel postprocess skipped for low-detail runtime mesh (source_vertices={} < {}), because this regime is prone to chair regressions (monochrome PBR + high boundary-hole ratio)",
            source_vertices, OVOXEL_MIN_SOURCE_VERTICES
        ));
    }
    None
}

fn run_ovoxel_postprocess(args: &Args, hook_path: &Path, output_glb: &Path) -> Result<(), String> {
    if !hook_path.exists() {
        return Err(format!(
            "ovoxel postprocess input hook missing: {}",
            hook_path.display()
        ));
    }
    let script = args
        .ovoxel_postprocess_script
        .clone()
        .unwrap_or_else(default_ovoxel_postprocess_script_path);
    if !script.exists() {
        return Err(format!(
            "ovoxel postprocess script missing: {}",
            script.display()
        ));
    }
    let mut cmd = Command::new(&args.ovoxel_python_bin);
    cmd.arg(script.as_path())
        .arg("--hook")
        .arg(hook_path)
        .arg("--output")
        .arg(output_glb)
        .arg("--decimation-target")
        .arg(args.ovoxel_decimation_target.to_string())
        .arg("--texture-size")
        .arg(args.ovoxel_texture_size.to_string())
        .arg("--remesh-band")
        .arg(args.ovoxel_remesh_band.to_string())
        .arg("--remesh-project")
        .arg(args.ovoxel_remesh_project.to_string());
    if args.ovoxel_extension_webp {
        cmd.arg("--extension-webp");
    }
    let output = cmd
        .output()
        .map_err(|err| format!("failed to run o_voxel postprocess command: {err}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(output.stderr.as_slice())
            .trim()
            .to_string();
        let stdout = String::from_utf8_lossy(output.stdout.as_slice())
            .trim()
            .to_string();
        return Err(format!(
            "o_voxel postprocess failed (status={}): stderr='{}' stdout='{}'",
            output
                .status
                .code()
                .map(|code| code.to_string())
                .unwrap_or_else(|| "terminated_by_signal".to_string()),
            stderr,
            stdout
        ));
    }
    Ok(())
}

fn default_ovoxel_postprocess_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tool/trellis2_postprocess_from_hook.py")
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

    #[test]
    fn accepts_ovoxel_postprocess_flags() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input",
            "input.png",
            "--output",
            "out.glb",
            "--hook-output",
            "out_hook.safetensors",
            "--ovoxel-postprocess-from-hook",
            "--ovoxel-python-bin",
            "python3",
            "--ovoxel-decimation-target",
            "1000000",
            "--ovoxel-texture-size",
            "4096",
            "--ovoxel-remesh-band",
            "1.0",
            "--ovoxel-remesh-project",
            "0.0",
        ]);
        assert!(args.ovoxel_postprocess_from_hook);
        assert_eq!(args.ovoxel_python_bin, "python3");
        assert_eq!(args.ovoxel_decimation_target, 1_000_000);
        assert_eq!(args.ovoxel_texture_size, 4096);
        assert_eq!(args.ovoxel_remesh_band, 1.0);
        assert_eq!(args.ovoxel_remesh_project, 0.0);
    }

    #[test]
    fn accepts_runtime_debug_and_telemetry_flags() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input",
            "input.png",
            "--runtime-stage-debug",
            "--runtime-attention-debug",
            "--runtime-decoder-conv-telemetry",
        ]);
        assert!(args.runtime_stage_debug);
        assert!(args.runtime_attention_debug);
        assert!(args.runtime_decoder_conv_telemetry);
    }

    #[test]
    fn accepts_sampler_step_overrides() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input",
            "input.png",
            "--sparse-steps",
            "2",
            "--shape-steps",
            "3",
            "--texture-steps",
            "4",
        ]);
        assert_eq!(args.sparse_steps, Some(2));
        assert_eq!(args.shape_steps, Some(3));
        assert_eq!(args.tex_steps, Some(4));
    }

    #[test]
    fn strict_benchmark_rejects_stage_debug_probe_path() {
        let args = Args::parse_from([
            "trellis2_run",
            "--input",
            "input.png",
            "--strict-benchmark",
            "--runtime-stage-debug",
        ]);
        let err = validate_benchmark_args(&args)
            .expect_err("strict benchmark should reject stage-debug probe instrumentation");
        assert!(
            err.contains("--runtime-stage-debug"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn ovoxel_skip_reason_rejects_low_detail_meshes() {
        let reason = ovoxel_skip_reason(907_499)
            .expect("low-detail source mesh should skip o_voxel postprocess");
        assert!(
            reason.contains("source_vertices=907499"),
            "unexpected skip reason: {reason}"
        );
    }

    #[test]
    fn ovoxel_skip_reason_allows_dense_meshes() {
        assert!(
            ovoxel_skip_reason(4_497_407).is_none(),
            "dense source mesh should allow o_voxel postprocess"
        );
    }
}

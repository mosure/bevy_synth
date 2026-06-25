use std::path::{Path, PathBuf};
use std::time::Instant;

use burn_synth::{
    AssetBatchItem, AssetBatchRequest, DinoBackend, ForegroundRequest, ImageSource, ModelSelection,
    ProgressVerbosity, RuntimeBatchPolicy, RuntimeBatchStats, RuntimeConfig,
    RuntimeProgressObserver, SplatRequest, SynthRuntime, SynthesisAsset,
    default_log_progress_callback,
    quality::{DEFAULT_TRELLIS_TARGET_FACES, DEFAULT_TRIPOSG_TARGET_FACES, RuntimeQualityPreset},
    triposplat::TripoSplatBurnpackPrecision,
    write_glb_mesh,
};
use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use serde_json::json;

#[derive(Parser, Debug)]
#[command(
    name = "burn_synth",
    version,
    about = "burn_synth CLI for foreground extraction and image-to-mesh synthesis"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[arg(long, value_enum, default_value_t = CliForegroundModel::Rmbg14)]
    rmbg_model: CliForegroundModel,

    #[arg(long, value_enum, value_delimiter = ',')]
    synthesis_models: Vec<CliSynthesisModel>,

    #[arg(long, value_enum, default_value_t = CliBackend::Wgpu)]
    backend: CliBackend,

    #[arg(long)]
    weights_root: Option<PathBuf>,

    #[arg(long)]
    trellis_weights_root: Option<PathBuf>,

    #[arg(long)]
    triposplat_weights_root: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = CliTripoSplatWeightsPrecision::F32)]
    triposplat_weights_precision: CliTripoSplatWeightsPrecision,

    #[arg(long)]
    trellis_image_large_root: Option<PathBuf>,

    #[arg(long)]
    trellis_python_bin: Option<PathBuf>,

    #[arg(long)]
    trellis_bridge_script: Option<PathBuf>,

    #[arg(long)]
    trellis_noise_overrides_hook: Option<PathBuf>,

    #[arg(long)]
    trellis_max_sparse_coords: Option<usize>,

    /// Trellis PBR texture size for Rust/Burn o_voxel GLB export. Use 0 for runtime default.
    #[arg(long)]
    trellis_pbr_texture_size: Option<usize>,

    /// Enable Trellis PBR texture baking through the Rust/Burn o_voxel export path.
    #[arg(long, default_value_t = false, action = ArgAction::Set)]
    trellis_pbr: bool,

    #[arg(long, value_enum, default_value_t = CliTrellisQuality::Low)]
    trellis_quality: CliTrellisQuality,

    #[arg(long, value_enum, default_value_t = CliTrellisComputeProfile::ReferenceF32)]
    trellis_compute_profile: CliTrellisComputeProfile,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = CliQuality::Balanced)]
    quality: CliQuality,

    /// TripoSplat profile (low, balanced, high). Individual TripoSplat flags override this preset.
    #[arg(long, value_enum, default_value_t = CliTripoSplatProfile::Balanced)]
    triposplat_profile: CliTripoSplatProfile,

    #[arg(long)]
    bg_weights_root: Option<PathBuf>,

    #[arg(long)]
    num_steps: Option<usize>,

    #[arg(long)]
    num_tokens: Option<usize>,

    #[arg(long)]
    guidance_scale: Option<f32>,

    #[arg(long)]
    triposplat_shift: Option<f32>,

    /// Target Gaussian count(s) for TripoSplat. Values are rounded to a multiple of 32.
    #[arg(long, value_delimiter = ',')]
    gaussians: Vec<usize>,

    /// Alpha matte erosion radius for TripoSplat preprocessing.
    #[arg(long)]
    triposplat_erode_radius: Option<usize>,

    #[arg(long)]
    seed: Option<u64>,

    /// DINO backend (auto, cpu, gpu).
    #[arg(long, value_enum, default_value_t = CliDinoBackend::Auto)]
    dino_backend: CliDinoBackend,

    /// Target face count for mesh decimation. Use 0 to disable.
    /// Defaults to 10,000.
    #[arg(long)]
    faces: Option<usize>,

    #[arg(long)]
    flash_octree_depth: Option<usize>,

    #[arg(long)]
    flash_num_chunks: Option<usize>,

    #[arg(long)]
    flash_min_resolution: Option<usize>,

    #[arg(long)]
    flash_mini_grid_num: Option<usize>,

    #[arg(long, value_enum, default_value_t = CliProgress::Steps, global = true)]
    progress: CliProgress,

    #[arg(long, default_value_t = 1, global = true)]
    progress_every: usize,

    /// Batch chunk size for repeated --input values. Use "auto" to bound chunks by detected free VRAM.
    #[arg(long, default_value = "auto", value_parser = parse_cli_batch_size, global = true)]
    batch_size: CliBatchSize,

    /// Explicit VRAM budget in MB for auto batch planning.
    #[arg(long, global = true)]
    batch_vram_mb: Option<u64>,

    /// Optional machine-readable batch report path.
    #[arg(long, global = true)]
    batch_report_json: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Extract an RGBA foreground image.
    Foreground {
        /// Input image path. Repeat --input to process multiple images with one loaded runtime.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,

        /// Output image path. Defaults to `<input>_foreground.png`.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run image-to-mesh synthesis and write a GLB.
    Mesh {
        /// Input image path. Repeat --input to process multiple images with one loaded runtime.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,

        /// Output mesh path. Defaults to `<input>_mesh.glb`.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Skip model inference and emit a canonical cube.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
    /// Run image-to-splat synthesis and write a .splat or .ply file.
    Splat {
        /// Input image path. Repeat --input to process multiple images with one loaded runtime.
        #[arg(long = "input", required = true)]
        inputs: Vec<PathBuf>,

        /// Output splat path. Defaults to `<input>_splat.splat`.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Skip model inference and emit a tiny canonical debug splat cloud.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CliBatchSize {
    Auto,
    Fixed(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliForegroundModel {
    Rmbg14,
    Rmbg2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliSynthesisModel {
    Triposg,
    Trellis,
    Triposplat,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliBackend {
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum CliTripoSplatWeightsPrecision {
    Auto,
    F16,
    F32,
}

#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum CliTripoSplatProfile {
    Low,
    Balanced,
    High,
    Custom,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliDinoBackend {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliTrellisQuality {
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "kebab-case")]
enum CliTrellisComputeProfile {
    ReferenceF32,
    WgpuFastMixedF16,
    WgpuFastSparseSelfF16,
    WgpuFastSparseCrossF16,
    WgpuFastF16Tail1F32,
    WgpuFastF16Tail2F32,
    WgpuFastF16Tail4F32,
    WgpuFastF16Tail6F32,
    WgpuFastF16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliQuality {
    Fast,
    Balanced,
    Full,
}

impl CliQuality {
    fn defaults(self) -> burn_synth::quality::RuntimeQualityDefaults {
        RuntimeQualityPreset::from(self).defaults()
    }
}

impl From<CliQuality> for RuntimeQualityPreset {
    fn from(value: CliQuality) -> Self {
        match value {
            CliQuality::Fast => Self::Fast,
            CliQuality::Balanced => Self::Balanced,
            CliQuality::Full => Self::Full,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliProgress {
    Off,
    Stages,
    Steps,
}

fn main() {
    init_logging();
    let cli = Cli::parse();
    if let Err(err) = run_with_large_stack(cli) {
        eprintln!("burn_synth error: {err}");
        std::process::exit(1);
    }
}

fn run_with_large_stack(cli: Cli) -> Result<(), String> {
    const STACK_SIZE_BYTES: usize = 256 * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .name("burn_synth_main".to_string())
        .stack_size(STACK_SIZE_BYTES)
        .spawn(move || run(cli))
        .map_err(|err| format!("failed to start burn_synth worker thread: {err}"))?;
    match handle.join() {
        Ok(result) => result,
        Err(payload) => {
            let message = if let Some(text) = payload.downcast_ref::<&str>() {
                (*text).to_string()
            } else if let Some(text) = payload.downcast_ref::<String>() {
                text.clone()
            } else {
                "unknown panic payload".to_string()
            };
            Err(format!("burn_synth worker thread panicked: {message}"))
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let quality_defaults = cli.quality.defaults();
    let synthesis_models = resolve_cli_synthesis_models(&cli)?;
    ensure_requested_models_supported(&synthesis_models)?;
    let uses_triposplat = matches!(cli.command, Command::Splat { .. })
        || synthesis_models
            .first()
            .is_some_and(|model| matches!(model, CliSynthesisModel::Triposplat));
    let triposplat_profile_settings =
        burn_synth::triposplat::TripoSplatProfile::from(cli.triposplat_profile).settings();
    let uses_trellis = synthesis_models
        .iter()
        .any(|model| matches!(model, CliSynthesisModel::Trellis));
    let target_faces = match cli.faces {
        Some(0) => None,
        Some(value) => Some(value),
        None if uses_trellis => Some(DEFAULT_TRELLIS_TARGET_FACES),
        None => Some(DEFAULT_TRIPOSG_TARGET_FACES),
    };
    let triposplat_counts =
        normalize_cli_gaussian_counts(&cli.gaussians, triposplat_profile_settings.num_gaussians)?;
    let triposplat_num_steps = cli.num_steps.unwrap_or(triposplat_profile_settings.steps);
    let triposplat_guidance_scale = cli
        .guidance_scale
        .unwrap_or(triposplat_profile_settings.guidance_scale);
    let default_runtime_config = RuntimeConfig::default();
    let mut runtime_config = RuntimeConfig {
        model_selection: ModelSelection::new(
            synthesis_models.iter().copied().map(Into::into),
            cli.rmbg_model.into(),
        ),
        backend: cli.backend.into(),
        weights_root: cli.weights_root,
        trellis_weights_root: cli.trellis_weights_root,
        triposplat_weights_root: cli.triposplat_weights_root,
        triposplat_weights_precision: cli.triposplat_weights_precision.into(),
        trellis_image_large_root: cli.trellis_image_large_root,
        trellis_python_bin: cli.trellis_python_bin,
        trellis_bridge_script: cli.trellis_bridge_script,
        trellis_noise_overrides_hook: cli.trellis_noise_overrides_hook,
        trellis_max_sparse_coords: cli.trellis_max_sparse_coords,
        trellis_pbr_texture_size: cli
            .trellis_pbr_texture_size
            .filter(|size| *size > 0)
            .or(default_runtime_config.trellis_pbr_texture_size),
        trellis_pbr_enabled: cli.trellis_pbr,
        trellis_quality: cli.trellis_quality.into(),
        trellis_compute_profile: cli.trellis_compute_profile.into(),
        bg_weights_root: cli.bg_weights_root,
        num_steps: cli.num_steps.unwrap_or(if uses_triposplat {
            triposplat_profile_settings.steps
        } else {
            quality_defaults.num_steps
        }),
        num_tokens: cli.num_tokens.unwrap_or(quality_defaults.num_tokens),
        guidance_scale: cli.guidance_scale.unwrap_or(if uses_triposplat {
            triposplat_profile_settings.guidance_scale
        } else {
            quality_defaults.guidance_scale
        }),
        triposplat_num_steps,
        triposplat_guidance_scale,
        triposplat_shift: cli.triposplat_shift.unwrap_or(3.0),
        triposplat_num_gaussians: triposplat_counts[0],
        triposplat_erode_radius: cli.triposplat_erode_radius.unwrap_or(1),
        seed: cli.seed.or(default_runtime_config.seed),
        dino_backend: cli.dino_backend.into(),
        target_faces,
        ..default_runtime_config
    };
    runtime_config.flash_extract.octree_depth = cli
        .flash_octree_depth
        .unwrap_or(quality_defaults.flash_octree_depth);
    runtime_config.flash_extract.num_chunks = cli
        .flash_num_chunks
        .unwrap_or(quality_defaults.flash_num_chunks);
    runtime_config.flash_extract.min_resolution = cli
        .flash_min_resolution
        .unwrap_or(quality_defaults.flash_min_resolution);
    runtime_config.flash_extract.mini_grid_num = cli
        .flash_mini_grid_num
        .unwrap_or(quality_defaults.flash_mini_grid_num);
    if !matches!(cli.progress, CliProgress::Off) {
        runtime_config.progress = RuntimeProgressObserver::with_callback(
            cli.progress.into(),
            cli.progress_every.max(1),
            default_log_progress_callback(),
        );
    }
    let mut runtime = SynthRuntime::new(runtime_config);
    let batch_policy = runtime_batch_policy(cli.batch_size, cli.batch_vram_mb);
    let batch_report_path = cli.batch_report_json.clone();
    let mut batch_report_entries = Vec::new();

    match cli.command {
        Command::Foreground { inputs, output } => {
            ensure_inputs_exist(inputs.as_slice())?;
            let input_count = inputs.len();
            for input in inputs.iter() {
                let command_start = Instant::now();
                let output = resolve_foreground_output_path(
                    output.as_deref(),
                    input.as_path(),
                    input_count,
                )?;
                let result = runtime
                    .extract_foreground(ForegroundRequest {
                        image: ImageSource::from_path(input.clone()),
                        model: Some(cli.rmbg_model.into()),
                    })
                    .map_err(|err| err.to_string())?;
                result
                    .image
                    .save(&output)
                    .map_err(|err| format!("failed to save {}: {err}", output.display()))?;
                println!(
                    "foreground saved: {} ({}x{}, model={}, total_ms={:.1})",
                    output.display(),
                    result.width,
                    result.height,
                    foreground_model_name(result.model),
                    command_start.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        Command::Mesh {
            inputs,
            output,
            dry_run,
        } => {
            ensure_inputs_exist(inputs.as_slice())?;
            let input_count = inputs.len();
            let output_paths = inputs
                .iter()
                .map(|input| {
                    resolve_glb_output_path(output.as_deref(), input.as_path(), input_count)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let batch_start = Instant::now();
            let batch = runtime
                .synthesize_assets_batch(AssetBatchRequest {
                    items: inputs
                        .iter()
                        .enumerate()
                        .map(|(index, input)| {
                            AssetBatchItem::new(
                                format!("mesh_{index}"),
                                ImageSource::from_path(input.clone()),
                            )
                        })
                        .collect(),
                    foreground_model: Some(cli.rmbg_model.into()),
                    synthesis_models: Some(
                        synthesis_models.iter().copied().map(Into::into).collect(),
                    ),
                    backend: Some(cli.backend.into()),
                    dry_run,
                    policy: batch_policy,
                })
                .map_err(|err| err.to_string())?;
            for (item, (input, output)) in batch
                .items
                .iter()
                .zip(inputs.iter().zip(output_paths.iter()))
            {
                let result = item.output.as_ref().map_err(|err| err.to_string())?;
                let mesh = match &result.asset {
                    SynthesisAsset::Mesh(mesh) => mesh,
                    SynthesisAsset::GaussianSplat(_) => {
                        return Err(
                            "internal output mismatch: mesh command returned Gaussian splats"
                                .to_string(),
                        );
                    }
                };
                write_glb_mesh(output.as_path(), mesh)?;
                println!(
                    "mesh saved: {} (vertices={}, faces={}, fg_model={}, synth_backend={}, backend={}, dry_run={}, chunk={}, item_ms={:.1})",
                    output.display(),
                    mesh.vertices.len(),
                    mesh.faces.len(),
                    foreground_model_name(result.foreground_model),
                    synthesis_model_name(result.synthesis_backend),
                    backend_name(result.backend),
                    dry_run,
                    item.chunk_index,
                    item.elapsed_ms
                );
                batch_report_entries.push(json!({
                    "command": "mesh",
                    "input": input.display().to_string(),
                    "output": output.display().to_string(),
                    "chunk_index": item.chunk_index,
                    "item_index": item.item_index,
                    "elapsed_ms": item.elapsed_ms,
                    "vertices": mesh.vertices.len(),
                    "faces": mesh.faces.len(),
                    "backend": backend_name(result.backend),
                    "synthesis_backend": synthesis_model_name(result.synthesis_backend),
                    "dry_run": dry_run,
                }));
            }
            push_batch_stats_report(&mut batch_report_entries, "mesh", batch_start, &batch.stats);
        }
        Command::Splat {
            inputs,
            output,
            dry_run,
        } => {
            ensure_inputs_exist(inputs.as_slice())?;
            let input_count = inputs.len();
            if triposplat_counts.len() == 1 {
                let output_paths = inputs
                    .iter()
                    .map(|input| {
                        resolve_splat_output_paths(
                            output.as_deref(),
                            input.as_path(),
                            &triposplat_counts,
                            input_count,
                        )
                        .and_then(|paths| {
                            paths
                                .into_iter()
                                .next()
                                .ok_or_else(|| "internal empty splat output path list".to_string())
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let batch_start = Instant::now();
                let batch = runtime
                    .synthesize_assets_batch(AssetBatchRequest {
                        items: inputs
                            .iter()
                            .enumerate()
                            .map(|(index, input)| {
                                AssetBatchItem::new(
                                    format!("splat_{index}"),
                                    ImageSource::from_path(input.clone()),
                                )
                            })
                            .collect(),
                        foreground_model: Some(cli.rmbg_model.into()),
                        synthesis_models: Some(
                            synthesis_models.iter().copied().map(Into::into).collect(),
                        ),
                        backend: Some(cli.backend.into()),
                        dry_run,
                        policy: batch_policy,
                    })
                    .map_err(|err| err.to_string())?;
                for (item, (input, output)) in batch
                    .items
                    .iter()
                    .zip(inputs.iter().zip(output_paths.iter()))
                {
                    let result = item.output.as_ref().map_err(|err| err.to_string())?;
                    let splats = match &result.asset {
                        SynthesisAsset::GaussianSplat(splats) => splats,
                        SynthesisAsset::Mesh(_) => {
                            return Err(
                                "internal TripoSplat output mismatch: splat command returned a mesh"
                                    .to_string(),
                            );
                        }
                    };
                    write_splat_output(output.as_path(), splats)?;
                    println!(
                        "splat saved: {} (gaussians={}, splats={}, fg_model={}, synth_backend={}, backend={}, dry_run={}, chunk={}, item_ms={:.1})",
                        output.display(),
                        triposplat_counts[0],
                        splats.len(),
                        foreground_model_name(result.foreground_model),
                        synthesis_model_name(result.synthesis_backend),
                        backend_name(result.backend),
                        dry_run,
                        item.chunk_index,
                        item.elapsed_ms
                    );
                    batch_report_entries.push(json!({
                        "command": "splat",
                        "input": input.display().to_string(),
                        "output": output.display().to_string(),
                        "chunk_index": item.chunk_index,
                        "item_index": item.item_index,
                        "elapsed_ms": item.elapsed_ms,
                        "gaussians": triposplat_counts[0],
                        "splats": splats.len(),
                        "backend": backend_name(result.backend),
                        "synthesis_backend": synthesis_model_name(result.synthesis_backend),
                        "dry_run": dry_run,
                    }));
                }
                push_batch_stats_report(
                    &mut batch_report_entries,
                    "splat",
                    batch_start,
                    &batch.stats,
                );
            } else {
                let batch_start = Instant::now();
                for input in inputs.iter() {
                    let command_start = Instant::now();
                    let result = runtime
                        .synthesize_splats(SplatRequest {
                            image: ImageSource::from_path(input.clone()),
                            foreground_model: Some(cli.rmbg_model.into()),
                            backend: Some(cli.backend.into()),
                            num_gaussians: triposplat_counts.clone(),
                            dry_run,
                        })
                        .map_err(|err| err.to_string())?;
                    let outputs = resolve_splat_output_paths(
                        output.as_deref(),
                        input.as_path(),
                        &result.num_gaussians,
                        input_count,
                    )?;
                    if outputs.len() != result.splats.len() {
                        return Err(format!(
                            "internal TripoSplat output mismatch: {} paths for {} splat clouds",
                            outputs.len(),
                            result.splats.len()
                        ));
                    }
                    let total_ms = command_start.elapsed().as_secs_f64() * 1000.0;
                    for ((output, splats), count) in outputs
                        .iter()
                        .zip(result.splats.iter())
                        .zip(result.num_gaussians.iter())
                    {
                        write_splat_output(output.as_path(), splats)?;
                        println!(
                            "splat saved: {} (gaussians={}, splats={}, fg_model={}, synth_backend={}, backend={}, dry_run={}, total_ms={:.1})",
                            output.display(),
                            count,
                            splats.len(),
                            foreground_model_name(result.foreground_model),
                            synthesis_model_name(result.synthesis_backend),
                            backend_name(result.backend),
                            dry_run,
                            total_ms
                        );
                        batch_report_entries.push(json!({
                            "command": "splat",
                            "input": input.display().to_string(),
                            "output": output.display().to_string(),
                            "elapsed_ms": total_ms,
                            "gaussians": count,
                            "splats": splats.len(),
                            "backend": backend_name(result.backend),
                            "synthesis_backend": synthesis_model_name(result.synthesis_backend),
                            "dry_run": dry_run,
                            "execution_mode": "serial_reuse_multi_density",
                        }));
                    }
                }
                batch_report_entries.push(json!({
                    "command": "splat",
                    "type": "batch_stats",
                    "total_items": inputs.len(),
                    "chunk_size": 1,
                    "chunks": inputs.len(),
                    "execution_mode": "serial_reuse_multi_density",
                    "vram_budget_mb": null,
                    "estimated_item_mb": null,
                    "runtime_elapsed_ms": null,
                    "wall_elapsed_ms": batch_start.elapsed().as_secs_f64() * 1000.0,
                }));
            }
        }
    }

    if let Some(report_path) = batch_report_path.as_deref() {
        write_batch_report(report_path, &batch_report_entries)?;
    }

    Ok(())
}

fn parse_cli_batch_size(value: &str) -> Result<CliBatchSize, String> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(CliBatchSize::Auto);
    }
    let size = value
        .parse::<usize>()
        .map_err(|err| format!("invalid batch size '{value}': {err}"))?;
    if size == 0 {
        return Err("batch size must be 'auto' or a positive integer".to_string());
    }
    Ok(CliBatchSize::Fixed(size))
}

fn runtime_batch_policy(
    batch_size: CliBatchSize,
    vram_budget_mb: Option<u64>,
) -> RuntimeBatchPolicy {
    RuntimeBatchPolicy {
        max_items: match batch_size {
            CliBatchSize::Auto => None,
            CliBatchSize::Fixed(size) => Some(size),
        },
        vram_budget_mb,
        ..RuntimeBatchPolicy::default()
    }
}

fn push_batch_stats_report(
    entries: &mut Vec<serde_json::Value>,
    command: &str,
    start: Instant,
    stats: &RuntimeBatchStats,
) {
    entries.push(json!({
        "command": command,
        "type": "batch_stats",
        "total_items": stats.total_items,
        "chunk_size": stats.chunk_size,
        "chunks": stats.chunks,
        "execution_mode": stats.execution_mode.as_str(),
        "vram_budget_mb": stats.vram_budget_mb,
        "estimated_item_mb": stats.estimated_item_mb,
        "runtime_elapsed_ms": stats.elapsed_ms,
        "wall_elapsed_ms": start.elapsed().as_secs_f64() * 1000.0,
    }));
}

fn write_batch_report(path: &Path, entries: &[serde_json::Value]) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
    }
    let report = json!({
        "schema": "burn_synth.batch_report.v1",
        "entries": entries,
    });
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed to serialize batch report: {err}"))?;
    std::fs::write(path, bytes)
        .map_err(|err| format!("failed to write batch report {}: {err}", path.display()))
}

fn init_logging() {
    let mut builder = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("burn_synth=info"),
    );
    builder.format_timestamp_millis();
    let _ = builder.try_init();
}

fn ensure_exists(path: &Path) -> Result<(), String> {
    if path.exists() {
        Ok(())
    } else {
        Err(format!("path does not exist: {}", path.display()))
    }
}

fn ensure_inputs_exist(inputs: &[PathBuf]) -> Result<(), String> {
    if inputs.is_empty() {
        return Err("at least one --input is required".to_string());
    }
    for input in inputs {
        ensure_exists(input.as_path())?;
    }
    Ok(())
}

fn normalize_cli_gaussian_counts(
    raw: &[usize],
    default_count: usize,
) -> Result<Vec<usize>, String> {
    let counts = if raw.is_empty() {
        vec![default_count]
    } else {
        raw.to_vec()
    };
    let counts = counts
        .into_iter()
        .map(burn_synth::triposplat::normalize_num_gaussians)
        .collect::<Result<Vec<_>, _>>()?;
    if counts.is_empty() {
        Err("at least one TripoSplat gaussian count is required".to_string())
    } else {
        Ok(counts)
    }
}

fn default_output_path(input: &Path, suffix: &str, ext: &str) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{suffix}.{ext}"))
}

fn output_dir_path_or_error(
    output: Option<&Path>,
    input: &Path,
    suffix: &str,
    ext: &str,
    input_count: usize,
) -> Result<Option<PathBuf>, String> {
    let Some(path) = output else {
        return Ok(None);
    };
    if path.extension().is_none() || path.is_dir() {
        let stem = input
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        return Ok(Some(path.join(format!("{stem}{suffix}.{ext}"))));
    }
    if input_count > 1 {
        return Err(format!(
            "--output must be a directory or extensionless output root when processing multiple inputs; got {}",
            path.display()
        ));
    }
    Ok(None)
}

fn resolve_foreground_output_path(
    output: Option<&Path>,
    input: &Path,
    input_count: usize,
) -> Result<PathBuf, String> {
    let Some(path) = output else {
        return Ok(default_output_path(input, "_foreground", "png"));
    };
    if path.is_dir() || (input_count > 1 && path.extension().is_none()) {
        let stem = input
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        return Ok(path.join(format!("{stem}_foreground.png")));
    }
    if input_count > 1 {
        return Err(format!(
            "--output must be a directory or extensionless output root when processing multiple inputs; got {}",
            path.display()
        ));
    }
    Ok(path.to_path_buf())
}

fn resolve_glb_output_path(
    output: Option<&Path>,
    input: &Path,
    input_count: usize,
) -> Result<PathBuf, String> {
    if let Some(path) = output_dir_path_or_error(output, input, "_mesh", "glb", input_count)? {
        return Ok(path);
    }
    let Some(path) = output else {
        return Ok(default_output_path(input, "_mesh", "glb"));
    };
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("glb"))
        .unwrap_or(false)
    {
        Ok(path.to_path_buf())
    } else {
        Ok(path.with_extension("glb"))
    }
}

fn resolve_splat_output_path(
    output: Option<&Path>,
    input: &Path,
    input_count: usize,
) -> Result<PathBuf, String> {
    if let Some(path) = output_dir_path_or_error(output, input, "_splat", "splat", input_count)? {
        return Ok(path);
    }
    let Some(path) = output else {
        return Ok(default_output_path(input, "_splat", "splat"));
    };
    let is_supported = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("splat") || value.eq_ignore_ascii_case("ply"))
        .unwrap_or(false);
    if is_supported {
        Ok(path.to_path_buf())
    } else {
        Ok(path.with_extension("splat"))
    }
}

fn resolve_splat_output_paths(
    output: Option<&Path>,
    input: &Path,
    counts: &[usize],
    input_count: usize,
) -> Result<Vec<PathBuf>, String> {
    let base = resolve_splat_output_path(output, input, input_count)?;
    if counts.len() <= 1 {
        return Ok(vec![base]);
    }
    Ok(counts
        .iter()
        .map(|count| splat_output_with_count(base.as_path(), *count))
        .collect())
}

fn splat_output_with_count(path: &Path, count: usize) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("splat");
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("splat");
    parent.join(format!("{stem}_{count}.{ext}"))
}

fn write_splat_output(
    path: &Path,
    splats: &burn_synth::triposplat::GaussianSplatCloud,
) -> Result<(), String> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("ply"))
        .unwrap_or(false)
    {
        splats.write_ply(path)
    } else {
        splats.write_splat(path)
    }
}

fn resolve_cli_synthesis_models(cli: &Cli) -> Result<Vec<CliSynthesisModel>, String> {
    let explicit_models = !cli.synthesis_models.is_empty();
    let models = if explicit_models {
        sanitize_synthesis_models(cli.synthesis_models.clone())
    } else if matches!(cli.command, Command::Splat { .. }) {
        vec![CliSynthesisModel::Triposplat]
    } else {
        vec![CliSynthesisModel::Triposg]
    };
    if matches!(cli.command, Command::Splat { .. })
        && models.as_slice() != [CliSynthesisModel::Triposplat]
    {
        return Err(
            "splat command requires --synthesis-models triposplat or no --synthesis-models flag"
                .to_string(),
        );
    }
    Ok(models)
}

fn sanitize_synthesis_models(models: Vec<CliSynthesisModel>) -> Vec<CliSynthesisModel> {
    let mut out = Vec::new();
    for model in models {
        if !out.contains(&model) {
            out.push(model);
        }
    }
    if out.is_empty() {
        out.push(CliSynthesisModel::Triposg);
    }
    out
}

#[cfg(feature = "trellis")]
fn ensure_requested_models_supported(_models: &[CliSynthesisModel]) -> Result<(), String> {
    Ok(())
}

#[cfg(not(feature = "trellis"))]
fn ensure_requested_models_supported(models: &[CliSynthesisModel]) -> Result<(), String> {
    if models
        .iter()
        .any(|model| matches!(model, CliSynthesisModel::Trellis))
    {
        return Err(
            "trellis synthesis model requested, but this build does not enable burn_synth feature `trellis`"
                .to_string(),
        );
    }

    Ok(())
}

fn foreground_model_name(model: burn_synth::ForegroundModel) -> &'static str {
    match model {
        burn_synth::ForegroundModel::Rmbg14 => "rmbg14",
        burn_synth::ForegroundModel::Rmbg2 => "rmbg2",
    }
}

fn synthesis_model_name(model: burn_synth::SynthesisModel) -> &'static str {
    match model {
        burn_synth::SynthesisModel::Triposg => "triposg",
        burn_synth::SynthesisModel::Trellis => "trellis",
        burn_synth::SynthesisModel::Triposplat => "triposplat",
    }
}

fn backend_name(backend: burn_synth::InferenceBackend) -> &'static str {
    match backend {
        burn_synth::InferenceBackend::Cpu => "cpu",
        burn_synth::InferenceBackend::Wgpu => "wgpu",
        burn_synth::InferenceBackend::Cuda => "cuda",
    }
}

impl From<CliForegroundModel> for burn_synth::ForegroundModel {
    fn from(value: CliForegroundModel) -> Self {
        match value {
            CliForegroundModel::Rmbg14 => Self::Rmbg14,
            CliForegroundModel::Rmbg2 => Self::Rmbg2,
        }
    }
}

impl From<CliSynthesisModel> for burn_synth::SynthesisModel {
    fn from(value: CliSynthesisModel) -> Self {
        match value {
            CliSynthesisModel::Triposg => Self::Triposg,
            CliSynthesisModel::Trellis => Self::Trellis,
            CliSynthesisModel::Triposplat => Self::Triposplat,
        }
    }
}

impl From<CliBackend> for burn_synth::InferenceBackend {
    fn from(value: CliBackend) -> Self {
        match value {
            CliBackend::Cpu => Self::Cpu,
            CliBackend::Wgpu => Self::Wgpu,
            CliBackend::Cuda => Self::Cuda,
        }
    }
}

impl From<CliTripoSplatWeightsPrecision> for Option<TripoSplatBurnpackPrecision> {
    fn from(value: CliTripoSplatWeightsPrecision) -> Self {
        match value {
            CliTripoSplatWeightsPrecision::Auto => None,
            CliTripoSplatWeightsPrecision::F16 => Some(TripoSplatBurnpackPrecision::F16),
            CliTripoSplatWeightsPrecision::F32 => Some(TripoSplatBurnpackPrecision::F32),
        }
    }
}

impl From<CliTripoSplatProfile> for burn_synth::triposplat::TripoSplatProfile {
    fn from(value: CliTripoSplatProfile) -> Self {
        match value {
            CliTripoSplatProfile::Low => Self::Low,
            CliTripoSplatProfile::Balanced => Self::Balanced,
            CliTripoSplatProfile::High => Self::High,
            CliTripoSplatProfile::Custom => Self::Custom,
        }
    }
}

impl From<CliDinoBackend> for DinoBackend {
    fn from(value: CliDinoBackend) -> Self {
        match value {
            CliDinoBackend::Auto => Self::Auto,
            CliDinoBackend::Cpu => Self::Cpu,
            CliDinoBackend::Gpu => Self::Gpu,
        }
    }
}

impl From<CliTrellisQuality> for burn_synth::TrellisQuality {
    fn from(value: CliTrellisQuality) -> Self {
        match value {
            CliTrellisQuality::Low => Self::Low,
            CliTrellisQuality::Medium => Self::Medium,
            CliTrellisQuality::High => Self::High,
        }
    }
}

impl From<CliTrellisComputeProfile> for burn_synth::TrellisComputeProfile {
    fn from(value: CliTrellisComputeProfile) -> Self {
        match value {
            CliTrellisComputeProfile::ReferenceF32 => Self::ReferenceF32,
            CliTrellisComputeProfile::WgpuFastMixedF16 => Self::WgpuFastMixedF16,
            CliTrellisComputeProfile::WgpuFastSparseSelfF16 => Self::WgpuFastSparseSelfF16,
            CliTrellisComputeProfile::WgpuFastSparseCrossF16 => Self::WgpuFastSparseCrossF16,
            CliTrellisComputeProfile::WgpuFastF16Tail1F32 => Self::WgpuFastF16Tail1F32,
            CliTrellisComputeProfile::WgpuFastF16Tail2F32 => Self::WgpuFastF16Tail2F32,
            CliTrellisComputeProfile::WgpuFastF16Tail4F32 => Self::WgpuFastF16Tail4F32,
            CliTrellisComputeProfile::WgpuFastF16Tail6F32 => Self::WgpuFastF16Tail6F32,
            CliTrellisComputeProfile::WgpuFastF16 => Self::WgpuFastF16,
        }
    }
}

impl From<CliProgress> for ProgressVerbosity {
    fn from(value: CliProgress) -> Self {
        match value {
            CliProgress::Off => Self::Off,
            CliProgress::Stages => Self::Stages,
            CliProgress::Steps => Self::Steps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn full_quality_defaults_match_legacy_runtime_defaults() {
        let defaults = CliQuality::Full.defaults();
        assert_eq!(defaults.num_steps, 50);
        assert_eq!(defaults.num_tokens, 2048);
        assert_eq!(defaults.guidance_scale, 7.0);
        assert_eq!(defaults.flash_octree_depth, 9);
        assert_eq!(defaults.flash_num_chunks, 10_000);
        assert_eq!(defaults.flash_min_resolution, 63);
        assert_eq!(defaults.flash_mini_grid_num, 4);
    }

    #[test]
    fn cli_quality_defaults_to_balanced_for_mesh_subcommand() {
        let cli = Cli::parse_from(["burn_synth", "mesh", "--input", "input.png"]);
        assert_eq!(cli.quality, CliQuality::Balanced);
        assert_eq!(
            resolve_cli_synthesis_models(&cli).unwrap(),
            vec![CliSynthesisModel::Triposg]
        );
    }

    #[test]
    fn trellis_cli_defaults_to_reference_f32_profile() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--synthesis-models",
            "trellis",
            "mesh",
            "--input",
            "input.png",
        ]);
        assert_eq!(cli.trellis_quality, CliTrellisQuality::Low);
        assert!(!cli.trellis_pbr);
        assert_eq!(
            cli.trellis_compute_profile,
            CliTrellisComputeProfile::ReferenceF32
        );
    }

    #[test]
    fn mesh_cli_accepts_repeated_inputs_for_one_runtime_invocation() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--batch-size",
            "2",
            "mesh",
            "--input",
            "first.png",
            "--input",
            "second.png",
        ]);
        let Command::Mesh { inputs, .. } = cli.command else {
            panic!("expected mesh command");
        };
        assert_eq!(cli.batch_size, CliBatchSize::Fixed(2));
        assert_eq!(
            inputs,
            vec![PathBuf::from("first.png"), PathBuf::from("second.png")]
        );
    }

    #[test]
    fn cli_batch_size_defaults_to_auto() {
        let cli = Cli::parse_from(["burn_synth", "mesh", "--input", "input.png"]);
        assert_eq!(cli.batch_size, CliBatchSize::Auto);
        assert_eq!(parse_cli_batch_size("4").unwrap(), CliBatchSize::Fixed(4));
        assert!(parse_cli_batch_size("0").is_err());
    }

    #[test]
    fn explicit_flags_override_quality_preset_inputs() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--quality",
            "fast",
            "--num-steps",
            "18",
            "--flash-min-resolution",
            "47",
            "mesh",
            "--input",
            "input.png",
        ]);
        assert_eq!(cli.quality, CliQuality::Fast);
        assert_eq!(cli.num_steps, Some(18));
        assert_eq!(cli.flash_min_resolution, Some(47));
    }

    #[test]
    fn trellis_cli_accepts_pbr_and_quality_settings() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--synthesis-models",
            "trellis",
            "--trellis-quality",
            "high",
            "--trellis-pbr",
            "true",
            "--trellis-pbr-texture-size",
            "2048",
            "--faces",
            "500000",
            "mesh",
            "--input",
            "input.png",
        ]);
        assert_eq!(cli.trellis_quality, CliTrellisQuality::High);
        assert!(cli.trellis_pbr);
        assert_eq!(cli.trellis_pbr_texture_size, Some(2048));
        assert_eq!(cli.faces, Some(500_000));
        assert_eq!(
            resolve_cli_synthesis_models(&cli).unwrap(),
            vec![CliSynthesisModel::Trellis]
        );
    }

    #[test]
    fn splat_cli_accepts_comma_delimited_gaussian_counts() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--gaussians",
            "32769,65536",
            "splat",
            "--input",
            "input.png",
        ]);
        assert_eq!(cli.gaussians, vec![32_769, 65_536]);
        assert_eq!(
            normalize_cli_gaussian_counts(
                &cli.gaussians,
                burn_synth::triposplat::DEFAULT_NUM_GAUSSIANS
            )
            .unwrap(),
            vec![32_768, 65_536]
        );
        assert_eq!(
            resolve_cli_synthesis_models(&cli).unwrap(),
            vec![CliSynthesisModel::Triposplat]
        );
    }

    #[test]
    fn splat_cli_defaults_triposplat_precision_to_f32() {
        let cli = Cli::parse_from(["burn_synth", "splat", "--input", "input.png"]);
        assert_eq!(
            cli.triposplat_weights_precision,
            CliTripoSplatWeightsPrecision::F32
        );
    }

    #[test]
    fn splat_cli_profile_defaults_are_upstream_triposplat_settings() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--triposplat-profile",
            "low",
            "splat",
            "--input",
            "input.png",
        ]);
        let settings =
            burn_synth::triposplat::TripoSplatProfile::from(cli.triposplat_profile).settings();
        assert_eq!(settings.steps, 5);
        assert_eq!(settings.guidance_scale, 3.0);
        assert_eq!(settings.num_gaussians, 32_768);
        assert_eq!(
            normalize_cli_gaussian_counts(&cli.gaussians, settings.num_gaussians).unwrap(),
            vec![32_768]
        );
    }

    #[test]
    fn splat_cli_rejects_explicit_mesh_synthesis_model() {
        let cli = Cli::parse_from([
            "burn_synth",
            "--synthesis-models",
            "triposg",
            "splat",
            "--input",
            "input.png",
        ]);
        let err = resolve_cli_synthesis_models(&cli)
            .expect_err("splat command should reject mesh synthesis models");
        assert!(err.contains("synthesis-models triposplat"));
    }

    #[test]
    fn multi_splat_outputs_preserve_extension_with_count_suffixes() {
        let outputs = resolve_splat_output_paths(
            Some(Path::new("out/model.ply")),
            Path::new("input.png"),
            &[32_768, 65_536],
            1,
        )
        .unwrap();
        assert_eq!(
            outputs,
            vec![
                PathBuf::from("out/model_32768.ply"),
                PathBuf::from("out/model_65536.ply"),
            ]
        );
    }

    #[test]
    fn multi_input_output_file_is_rejected_to_avoid_overwrite() {
        let err = resolve_splat_output_paths(
            Some(Path::new("out/model.splat")),
            Path::new("input.png"),
            &[32_768],
            2,
        )
        .expect_err("file output should be ambiguous for multiple inputs");
        assert!(err.contains("--output must be a directory"));
    }

    #[test]
    fn multi_input_output_directory_derives_per_input_names() {
        let output = resolve_glb_output_path(Some(Path::new("out")), Path::new("chair.png"), 2)
            .expect("directory-like output should be valid for multiple inputs");
        assert_eq!(output, PathBuf::from("out/chair_mesh.glb"));
    }

    #[test]
    fn splat_cli_dry_run_writes_single_asset_output() {
        let root = unique_test_dir("splat_cli_dry_run");
        fs::create_dir_all(&root).expect("failed to create temp test directory");
        let input = root.join("input.png");
        let output = root.join("output.splat");
        fs::write(&input, b"not decoded in dry-run").expect("failed to write temp input");

        let cli = Cli::parse_from([
            "burn_synth",
            "--backend",
            "cpu",
            "--progress",
            "off",
            "--gaussians",
            "32768",
            "splat",
            "--input",
            input.to_str().expect("utf-8 input path"),
            "--output",
            output.to_str().expect("utf-8 output path"),
            "--dry-run",
        ]);

        run(cli).expect("splat dry-run CLI should write a Gaussian splat asset");
        let metadata = fs::metadata(&output).expect("splat output should exist");
        assert!(metadata.len() > 0);
        fs::remove_dir_all(root).expect("failed to remove temp test directory");
    }

    #[test]
    fn splat_cli_dry_run_processes_repeated_inputs_with_one_runtime() {
        let root = unique_test_dir("splat_cli_dry_run_batch");
        let output_root = root.join("out");
        let report = root.join("batch_report.json");
        fs::create_dir_all(&output_root).expect("failed to create temp test directory");
        let first = root.join("first.png");
        let second = root.join("second.png");
        fs::write(&first, b"not decoded in dry-run").expect("failed to write first temp input");
        fs::write(&second, b"not decoded in dry-run").expect("failed to write second temp input");

        let cli = Cli::parse_from([
            "burn_synth",
            "--backend",
            "cpu",
            "--progress",
            "off",
            "--batch-size",
            "2",
            "--batch-report-json",
            report.to_str().expect("utf-8 report path"),
            "--gaussians",
            "32768",
            "splat",
            "--input",
            first.to_str().expect("utf-8 first input path"),
            "--input",
            second.to_str().expect("utf-8 second input path"),
            "--output",
            output_root.to_str().expect("utf-8 output path"),
            "--dry-run",
        ]);

        run(cli).expect("splat dry-run CLI should process every input");
        for output in ["first_splat.splat", "second_splat.splat"] {
            let output = output_root.join(output);
            let metadata = fs::metadata(&output).expect("splat batch output should exist");
            assert!(metadata.len() > 0);
        }
        let report: serde_json::Value =
            serde_json::from_slice(&fs::read(&report).expect("batch report should be written"))
                .expect("batch report should parse");
        let entries = report["entries"].as_array().expect("entries array");
        assert!(
            entries
                .iter()
                .any(|entry| entry["type"].as_str() == Some("batch_stats")
                    && entry["chunk_size"].as_u64() == Some(2))
        );
        fs::remove_dir_all(root).expect("failed to remove temp test directory");
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn_synth_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }
}

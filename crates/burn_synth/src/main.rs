use std::path::{Path, PathBuf};

use burn_synth::{
    DinoBackend, ForegroundRequest, ImageSource, MeshRequest, ModelSelection, ProgressVerbosity,
    RuntimeConfig, RuntimeProgressObserver, SynthRuntime, default_log_progress_callback,
    write_glb_mesh,
};
use clap::{Parser, Subcommand, ValueEnum};

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

    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_values_t = [CliSynthesisModel::Triposg]
    )]
    synthesis_models: Vec<CliSynthesisModel>,

    #[arg(long, value_enum, default_value_t = CliBackend::Wgpu)]
    backend: CliBackend,

    #[arg(long)]
    weights_root: Option<PathBuf>,

    #[arg(long)]
    trellis_weights_root: Option<PathBuf>,

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

    #[arg(long, value_enum, default_value_t = CliTrellisQuality::Medium)]
    trellis_quality: CliTrellisQuality,

    /// Quality preset (fast, balanced, full). Individual flags override this preset.
    #[arg(long, value_enum, default_value_t = CliQuality::Balanced)]
    quality: CliQuality,

    #[arg(long)]
    bg_weights_root: Option<PathBuf>,

    #[arg(long)]
    num_steps: Option<usize>,

    #[arg(long)]
    num_tokens: Option<usize>,

    #[arg(long)]
    guidance_scale: Option<f32>,

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
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Extract an RGBA foreground image.
    Foreground {
        /// Input image path.
        #[arg(long)]
        input: PathBuf,

        /// Output image path. Defaults to `<input>_foreground.png`.
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Run image-to-mesh synthesis and write a GLB.
    Mesh {
        /// Input image path.
        #[arg(long)]
        input: PathBuf,

        /// Output mesh path. Defaults to `<input>_mesh.glb`.
        #[arg(long)]
        output: Option<PathBuf>,

        /// Skip model inference and emit a canonical cube.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
    },
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
#[value(rename_all = "lower")]
enum CliBackend {
    Cpu,
    Wgpu,
    Cuda,
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
#[value(rename_all = "lower")]
enum CliQuality {
    Fast,
    Balanced,
    Full,
}

#[derive(Clone, Copy, Debug)]
struct CliQualityDefaults {
    num_steps: usize,
    num_tokens: usize,
    guidance_scale: f32,
    flash_octree_depth: usize,
    flash_num_chunks: usize,
    flash_min_resolution: usize,
    flash_mini_grid_num: usize,
}

impl CliQuality {
    fn defaults(self) -> CliQualityDefaults {
        match self {
            Self::Fast => CliQualityDefaults {
                num_steps: 12,
                num_tokens: 512,
                guidance_scale: 7.0,
                flash_octree_depth: 7,
                flash_num_chunks: 4096,
                flash_min_resolution: 31,
                flash_mini_grid_num: 2,
            },
            Self::Balanced => CliQualityDefaults {
                num_steps: 20,
                num_tokens: 1024,
                guidance_scale: 7.0,
                flash_octree_depth: 8,
                flash_num_chunks: 8192,
                flash_min_resolution: 31,
                flash_mini_grid_num: 4,
            },
            Self::Full => CliQualityDefaults {
                num_steps: 50,
                num_tokens: 2048,
                guidance_scale: 7.0,
                flash_octree_depth: 9,
                flash_num_chunks: 10_000,
                flash_min_resolution: 63,
                flash_mini_grid_num: 4,
            },
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
    let synthesis_models = sanitize_synthesis_models(cli.synthesis_models);
    ensure_requested_models_supported(&synthesis_models)?;
    let target_faces = match cli.faces {
        Some(0) => None,
        Some(value) => Some(value),
        None => Some(10_000),
    };
    let mut runtime_config = RuntimeConfig {
        model_selection: ModelSelection::new(
            synthesis_models.iter().copied().map(Into::into),
            cli.rmbg_model.into(),
        ),
        backend: cli.backend.into(),
        weights_root: cli.weights_root,
        trellis_weights_root: cli.trellis_weights_root,
        trellis_image_large_root: cli.trellis_image_large_root,
        trellis_python_bin: cli.trellis_python_bin,
        trellis_bridge_script: cli.trellis_bridge_script,
        trellis_noise_overrides_hook: cli.trellis_noise_overrides_hook,
        trellis_max_sparse_coords: cli.trellis_max_sparse_coords,
        trellis_quality: cli.trellis_quality.into(),
        bg_weights_root: cli.bg_weights_root,
        num_steps: cli.num_steps.unwrap_or(quality_defaults.num_steps),
        num_tokens: cli.num_tokens.unwrap_or(quality_defaults.num_tokens),
        guidance_scale: cli
            .guidance_scale
            .unwrap_or(quality_defaults.guidance_scale),
        seed: cli.seed.or(RuntimeConfig::default().seed),
        dino_backend: cli.dino_backend.into(),
        target_faces,
        ..RuntimeConfig::default()
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

    match cli.command {
        Command::Foreground { input, output } => {
            ensure_exists(input.as_path())?;
            let output = output
                .unwrap_or_else(|| default_output_path(input.as_path(), "_foreground", "png"));
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
                "foreground saved: {} ({}x{}, model={})",
                output.display(),
                result.width,
                result.height,
                foreground_model_name(result.model)
            );
        }
        Command::Mesh {
            input,
            output,
            dry_run,
        } => {
            ensure_exists(input.as_path())?;
            let output = resolve_glb_output_path(output, input.as_path());
            let result = runtime
                .synthesize_mesh(MeshRequest {
                    image: ImageSource::from_path(input.clone()),
                    foreground_model: Some(cli.rmbg_model.into()),
                    synthesis_models: Some(
                        synthesis_models.iter().copied().map(Into::into).collect(),
                    ),
                    backend: Some(cli.backend.into()),
                    dry_run,
                })
                .map_err(|err| err.to_string())?;
            write_glb_mesh(output.as_path(), &result.mesh)?;
            println!(
                "mesh saved: {} (vertices={}, faces={}, fg_model={}, synth_backend={}, backend={}, dry_run={})",
                output.display(),
                result.mesh.vertices.len(),
                result.mesh.faces.len(),
                foreground_model_name(result.foreground_model),
                synthesis_model_name(result.synthesis_backend),
                backend_name(result.backend),
                dry_run
            );
        }
    }

    Ok(())
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

fn default_output_path(input: &Path, suffix: &str, ext: &str) -> PathBuf {
    let parent = input.parent().unwrap_or_else(|| Path::new("."));
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    parent.join(format!("{stem}{suffix}.{ext}"))
}

fn resolve_glb_output_path(output: Option<PathBuf>, input: &Path) -> PathBuf {
    let Some(path) = output else {
        return default_output_path(input, "_mesh", "glb");
    };
    if path.extension().is_none() || path.is_dir() {
        let stem = input
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("mesh");
        return path.join(format!("{stem}_mesh.glb"));
    }
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("glb"))
        .unwrap_or(false)
    {
        path
    } else {
        path.with_extension("glb")
    }
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
}

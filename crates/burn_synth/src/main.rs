use std::path::{Path, PathBuf};

use burn_synth::{
    ForegroundRequest, ImageSource, MeshRequest, ModelSelection, ProgressVerbosity, RuntimeConfig,
    RuntimeProgressObserver, SynthRuntime, default_log_progress_callback, write_glb_mesh,
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

    #[arg(long, value_enum, default_value_t = CliForegroundModel::Rmbg2)]
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

    #[arg(long, value_enum, default_value_t = CliTrellisQuality::Medium)]
    trellis_quality: CliTrellisQuality,

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
enum CliTrellisQuality {
    Low,
    Medium,
    High,
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
    if let Err(err) = run(cli) {
        eprintln!("burn_synth error: {err}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    let synthesis_models = sanitize_synthesis_models(cli.synthesis_models);
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
        trellis_quality: cli.trellis_quality.into(),
        bg_weights_root: cli.bg_weights_root,
        num_steps: cli.num_steps.unwrap_or(RuntimeConfig::default().num_steps),
        num_tokens: cli
            .num_tokens
            .unwrap_or(RuntimeConfig::default().num_tokens),
        guidance_scale: cli
            .guidance_scale
            .unwrap_or(RuntimeConfig::default().guidance_scale),
        seed: cli.seed.or(RuntimeConfig::default().seed),
        ..RuntimeConfig::default()
    };
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

impl From<CliTrellisQuality> for burn_synth::trellis::TrellisQuality {
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

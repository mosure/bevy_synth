#![recursion_limit = "256"]

use std::path::{Path, PathBuf};

use burn::backend::NdArray;
use burn_synth_import::io::ensure_parent_dir;
use burn_synth_import::layout::{burnpack_path, precision_label};
use burn_synth_import::parts::{apply_artifact_policy, remove_legacy_shard_artifacts_for_burnpack};
use burn_synth_import::plan::ArtifactPolicy;
use clap::{Parser, ValueEnum};

use burn_tripo::model::triposg::{
    dit::{TripoSGDiTConfig, import::import_triposg_dit_burnpack},
    image_encoder::import::{import_triposg_dinov2_burnpack, resolve_triposg_weights_root},
    vae::{TripoSGVaeConfig, import::import_triposg_vae_burnpack},
};

type CpuBackend = NdArray<f32>;
type GpuBackend = burn_wgpu::Wgpu;

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Quantization {
    F32,
    F16,
    Both,
}

impl Quantization {
    fn include_f32(self) -> bool {
        matches!(self, Self::F32 | Self::Both)
    }

    fn include_f16(self) -> bool {
        matches!(self, Self::F16 | Self::Both)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ArtifactPolicyArg {
    SingleFile,
    Parts,
}

impl ArtifactPolicyArg {
    fn into_policy(self, part_size_mib: u64) -> ArtifactPolicy {
        let part_size_mib = part_size_mib.max(1);
        match self {
            Self::SingleFile => ArtifactPolicy::SingleFile,
            Self::Parts => ArtifactPolicy::Parts { part_size_mib },
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    about = "Import TripoSG safetensors into Burnpack (.bpk) files",
    version
)]
struct Args {
    #[arg(long)]
    weights_root: Option<PathBuf>,

    #[arg(long)]
    overwrite: bool,

    #[arg(long, value_enum, default_value_t = Quantization::F32)]
    quantization: Quantization,

    #[arg(long, value_enum, default_value_t = ArtifactPolicyArg::Parts)]
    artifact_policy: ArtifactPolicyArg,

    #[arg(long, default_value_t = 64)]
    part_size_mib: u64,
}

struct TripoSources<'a> {
    vae: &'a Path,
    dit: &'a Path,
    dino: &'a Path,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let weights_root = args
        .weights_root
        .unwrap_or_else(resolve_triposg_weights_root);

    let vae_path = weights_root.join("vae/diffusion_pytorch_model.safetensors");
    let dit_path = weights_root.join("transformer/diffusion_pytorch_model.safetensors");
    let dino_path = weights_root.join("image_encoder_dinov2/model.safetensors");

    let vae_config_path = weights_root.join("vae/config.json");
    let vae_config = TripoSGVaeConfig::from_config_file(vae_config_path)
        .unwrap_or_else(|_| TripoSGVaeConfig::midi_3d());

    let dit_config_path = weights_root.join("transformer/config.json");
    let dit_config = TripoSGDiTConfig::from_config_file(dit_config_path).unwrap_or_else(|_| {
        if dit_path.exists() {
            TripoSGDiTConfig::triposg_pretrained()
        } else {
            TripoSGDiTConfig::midi_3d()
        }
    });

    if args.quantization.include_f32() {
        run_imports_with_backend::<CpuBackend>(
            false,
            args.overwrite,
            TripoSources {
                vae: &vae_path,
                dit: &dit_path,
                dino: &dino_path,
            },
            &vae_config,
            &dit_config,
            args.artifact_policy.into_policy(args.part_size_mib),
        )?;
    }
    if args.quantization.include_f16() {
        run_imports_with_backend::<GpuBackend>(
            true,
            args.overwrite,
            TripoSources {
                vae: &vae_path,
                dit: &dit_path,
                dino: &dino_path,
            },
            &vae_config,
            &dit_config,
            args.artifact_policy.into_policy(args.part_size_mib),
        )?;
    }

    Ok(())
}

fn run_imports_with_backend<B>(
    use_f16: bool,
    overwrite: bool,
    sources: TripoSources<'_>,
    vae_config: &TripoSGVaeConfig,
    dit_config: &TripoSGDiTConfig,
    artifact_policy: ArtifactPolicy,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: burn::tensor::backend::Backend,
    B::Device: Default,
{
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    import_if_needed(
        "VAE",
        sources.vae,
        use_f16,
        overwrite,
        artifact_policy,
        || import_triposg_vae_burnpack::<B>(vae_config, &device, sources.vae, use_f16),
    )?;

    import_if_needed(
        "DiT",
        sources.dit,
        use_f16,
        overwrite,
        artifact_policy,
        || import_triposg_dit_burnpack::<B>(dit_config, &device, sources.dit, use_f16),
    )?;

    import_if_needed(
        "DINOv2",
        sources.dino,
        use_f16,
        overwrite,
        artifact_policy,
        || import_triposg_dinov2_burnpack::<B>(&device, sources.dino, use_f16),
    )?;

    Ok(())
}

fn import_if_needed<F>(
    label: &str,
    weights_path: &Path,
    use_f16: bool,
    overwrite: bool,
    artifact_policy: ArtifactPolicy,
    import_fn: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<PathBuf, Box<dyn std::error::Error>>,
{
    let burnpack = burnpack_path(weights_path, use_f16);
    if !weights_path.exists() && !burnpack.exists() {
        return Err(format!("missing {label} weights at {}", weights_path.display()).into());
    }
    let precision = precision_label(use_f16);
    let output = if burnpack.exists() && (!overwrite || !weights_path.exists()) {
        println!(
            "[IMPORT] {label} ({precision}) burnpack already exists at {}, skipping.",
            burnpack.display()
        );
        burnpack.clone()
    } else {
        ensure_parent_dir(&burnpack)?;
        let output = import_fn()?;
        println!(
            "[IMPORT] {label} ({precision}) burnpack saved to {}",
            output.display()
        );
        output
    };

    if let Some(report) = apply_artifact_policy(&output, artifact_policy, overwrite)? {
        println!(
            "[IMPORT] {label} ({precision}) parts manifest: {} ({} parts, {:.1} MiB)",
            report.manifest_path.display(),
            report.part_paths.len(),
            report.total_bytes as f64 / (1024.0 * 1024.0),
        );
    }
    let removed_legacy = remove_legacy_shard_artifacts_for_burnpack(&output)?;
    if removed_legacy > 0 {
        println!(
            "[IMPORT] {label} ({precision}) removed {removed_legacy} legacy shard artifact(s)"
        );
    }
    Ok(())
}

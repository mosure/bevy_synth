use std::path::PathBuf;

use burn_dino::model::dinov3::{DinoV3Config, import::import_dinov3_burnpack_to_path};
use burn_flux::{Flux2VaeEncoderConfig, flux2_import::import_flux2_vae_encoder_burnpack_to_path};
use burn_synth_import::io::ensure_parent_dir;
use burn_synth_import::parts::write_burnpack_parts_for_wasm;
use burn_triposplat::artifact::{
    TRIPOSPLAT_ARTIFACTS, TripoSplatBurnpackPrecision, TripoSplatCheckpointLayout,
};
use burn_triposplat::import::{
    import_triposplat_decoder_burnpack_to_path, import_triposplat_flow_burnpack_to_path,
};
use burn_triposplat::{
    ElasticGaussianFixedlenDecoderConfig, LatentSeqMmFlowModelConfig,
    OctreeProbabilityFixedlenDecoderConfig,
};
use clap::{Parser, ValueEnum};

#[cfg(feature = "backend_wgpu")]
type ImportBackend = burn::backend::Wgpu<f32, i32, u32>;
#[cfg(not(feature = "backend_wgpu"))]
type ImportBackend = burn::backend::NdArray<f32>;

#[derive(Debug, Parser)]
#[command(
    about = "Import TripoSplat safetensors into BurnPack artifacts and optional wasm shards",
    version
)]
struct Args {
    /// Root containing upstream TripoSplat Hugging Face checkpoint files.
    #[arg(long)]
    source_root: PathBuf,

    /// Root containing or receiving TripoSplat BurnPack files.
    #[arg(long)]
    output_root: PathBuf,

    /// BurnPack precision to validate or shard.
    #[arg(long, value_enum, default_value_t = Precision::F16)]
    precision: Precision,

    /// Only validate source and output paths without importing safetensors.
    #[arg(long, default_value_t = false)]
    validate_only: bool,

    /// Overwrite existing BurnPack files.
    #[arg(long, default_value_t = false)]
    overwrite: bool,

    /// Write wasm .bpk.parts.json files for existing BurnPacks.
    #[arg(long, default_value_t = false)]
    parts: bool,

    /// Part size in MiB for wasm BurnPack sharding.
    #[arg(long, default_value_t = 64)]
    part_size_mib: u64,

    /// Overwrite existing parts manifests and part files.
    #[arg(long, default_value_t = false)]
    overwrite_parts: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Precision {
    F32,
    F16,
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let source_layout = TripoSplatCheckpointLayout::new(&args.source_root);
    source_layout.validate_sources()?;

    let precision = match args.precision {
        Precision::F32 => TripoSplatBurnpackPrecision::F32,
        Precision::F16 => TripoSplatBurnpackPrecision::F16,
    };

    let use_f16 = precision == TripoSplatBurnpackPrecision::F16;

    eprintln!("[triposplat_import] source layout is valid");
    for artifact in TRIPOSPLAT_ARTIFACTS {
        let source = artifact.source_path(&args.source_root);
        let burnpack = artifact.burnpack_path(&args.output_root, precision);
        eprintln!(
            "[triposplat_import] component={} source={} burnpack={}",
            artifact.component,
            source.display(),
            burnpack.display()
        );
    }

    if !args.validate_only {
        let imported = import_burnpacks(&args, precision, use_f16)?;
        eprintln!("[triposplat_import] imported or reused {imported} burnpack artifact(s)");
    } else {
        eprintln!("[triposplat_import] validation only; skipping safetensors-to-burnpack import");
    }

    if args.parts {
        let mut generated = 0usize;
        for artifact in TRIPOSPLAT_ARTIFACTS {
            if !artifact.is_triposplat_runtime_required() {
                eprintln!(
                    "[triposplat_import] skipping optional upstream component={} during TripoSplat runtime sharding",
                    artifact.component
                );
                continue;
            }
            let burnpack = artifact.burnpack_path(&args.output_root, precision);
            if !burnpack.exists() {
                return Err(format!(
                    "cannot shard missing BurnPack {}; Burn DINOv3/Flux2/flow/octree/Gaussian modules are present, but upstream safetensors-to-.bpk conversion has not produced this component yet",
                    burnpack.display()
                ));
            }
            if write_burnpack_parts_for_wasm(&burnpack, args.part_size_mib, args.overwrite_parts)?
                .is_some()
            {
                generated += 1;
            }
        }
        eprintln!("[triposplat_import] generated or refreshed {generated} parts manifest(s)");
    } else {
        eprintln!(
            "[triposplat_import] parts disabled; pass --parts to generate wasm .bpk.parts.json manifests"
        );
    }

    Ok(())
}

fn import_burnpacks(
    args: &Args,
    precision: TripoSplatBurnpackPrecision,
    use_f16: bool,
) -> Result<usize, String> {
    let device = <ImportBackend as burn::tensor::backend::BackendTypes>::Device::default();
    let mut imported = 0usize;
    for artifact in TRIPOSPLAT_ARTIFACTS {
        if !artifact.is_triposplat_runtime_required() {
            eprintln!(
                "[triposplat_import] source-only optional component={} path={}",
                artifact.component,
                artifact.source_path(&args.source_root).display()
            );
            continue;
        }
        let source = artifact.source_path(&args.source_root);
        let output = artifact.burnpack_path(&args.output_root, precision);
        if output.exists() && !args.overwrite {
            eprintln!(
                "[triposplat_import] component={} burnpack already exists at {}, skipping",
                artifact.component,
                output.display()
            );
            imported += 1;
            continue;
        }
        ensure_parent_dir(&output)
            .map_err(|err| format!("failed to create parent for {}: {err}", output.display()))?;
        match artifact.burnpack_stem {
            "dino_v3_vit_h" => {
                import_dinov3_burnpack_to_path::<ImportBackend>(
                    &device,
                    &source,
                    &output,
                    &DinoV3Config::vit_h_16_plus(None),
                    use_f16,
                )
                .map_err(|err| format!("failed to import DINOv3: {err}"))?;
            }
            "flux2_vae_encoder" => {
                import_flux2_vae_encoder_burnpack_to_path::<ImportBackend>(
                    &device,
                    &source,
                    &output,
                    &Flux2VaeEncoderConfig::flux2(),
                    use_f16,
                )
                .map_err(|err| format!("failed to import Flux2 VAE encoder: {err}"))?;
            }
            "triposplat_flow" => {
                import_triposplat_flow_burnpack_to_path::<ImportBackend>(
                    &device,
                    &source,
                    &output,
                    &LatentSeqMmFlowModelConfig::triposplat(),
                    use_f16,
                )
                .map_err(|err| format!("failed to import TripoSplat flow: {err}"))?;
            }
            "triposplat_vae_decoder" => {
                import_triposplat_decoder_burnpack_to_path::<ImportBackend>(
                    &device,
                    &source,
                    &output,
                    &OctreeProbabilityFixedlenDecoderConfig::triposplat(),
                    &ElasticGaussianFixedlenDecoderConfig::triposplat(),
                    use_f16,
                )
                .map_err(|err| format!("failed to import TripoSplat decoder: {err}"))?;
            }
            other => {
                return Err(format!(
                    "no TripoSplat importer registered for burnpack stem {other}"
                ));
            }
        }
        eprintln!(
            "[triposplat_import] component={} imported {}",
            artifact.component,
            output.display()
        );
        imported += 1;
    }
    Ok(imported)
}

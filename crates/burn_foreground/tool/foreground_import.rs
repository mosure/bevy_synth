#![recursion_limit = "256"]

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("foreground_import is unavailable on wasm32 targets");
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::{Path, PathBuf};

    use burn::backend::NdArray;
    use burn_synth_import::io::ensure_parent_dir;
    use burn_synth_import::layout::{burnpack_path, precision_label};
    use burn_synth_import::parts::{
        apply_artifact_policy, remove_legacy_shard_artifacts_for_burnpack,
    };
    use burn_synth_import::plan::ArtifactPolicy;
    use clap::{Parser, ValueEnum};

    use burn_foreground::rmbg2::import::{import_rmbg2_burnpack, resolve_rmbg2_weights_root};
    use burn_foreground::rmbg14::{
        RmbgConfig,
        import::{import_rmbg_burnpack, load_rmbg_config, resolve_rmbg_weights_root},
    };

    type CpuBackend = NdArray<f32>;
    type GpuBackend = burn_wgpu::Wgpu;

    #[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
    enum Quantization {
        F32,
        F16,
        Both,
    }

    #[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
    enum RmbgModel {
        Rmbg14,
        Rmbg2,
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
        about = "Import foreground (RMBG) model weights into Burnpack (.bpk) files",
        version
    )]
    struct Args {
        #[arg(long)]
        rmbg_root: Option<PathBuf>,

        #[arg(long)]
        rmbg14_root: Option<PathBuf>,

        #[arg(long)]
        rmbg2_root: Option<PathBuf>,

        #[arg(long, value_enum, default_value_t = RmbgModel::Rmbg2)]
        rmbg_model: RmbgModel,

        #[arg(long)]
        overwrite: bool,

        #[arg(long, value_enum, default_value_t = Quantization::F32)]
        quantization: Quantization,

        #[arg(long, value_enum, default_value_t = ArtifactPolicyArg::Parts)]
        artifact_policy: ArtifactPolicyArg,

        #[arg(long, default_value_t = 64)]
        part_size_mib: u64,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();
        let (rmbg14_root, rmbg2_root) = resolve_model_roots(&args)?;

        let rmbg14 = if let Some(root) = rmbg14_root.as_ref() {
            let weights = root.join("model.safetensors");
            let config = load_rmbg_config(root)?;
            Some((weights, config))
        } else {
            None
        };

        if args.quantization.include_f32() {
            run_imports_with_backend::<CpuBackend>(
                false,
                args.overwrite,
                rmbg14
                    .as_ref()
                    .map(|(weights, config)| (weights.as_path(), config)),
                rmbg2_root.as_deref(),
                args.artifact_policy.into_policy(args.part_size_mib),
            )?;
        }
        if args.quantization.include_f16() {
            run_imports_with_backend::<GpuBackend>(
                true,
                args.overwrite,
                rmbg14
                    .as_ref()
                    .map(|(weights, config)| (weights.as_path(), config)),
                rmbg2_root.as_deref(),
                args.artifact_policy.into_policy(args.part_size_mib),
            )?;
        }

        Ok(())
    }

    fn resolve_model_roots(args: &Args) -> Result<(Option<PathBuf>, Option<PathBuf>), String> {
        match args.rmbg_model {
            RmbgModel::Rmbg14 => {
                let root = args
                    .rmbg14_root
                    .clone()
                    .or_else(|| args.rmbg_root.clone())
                    .unwrap_or_else(resolve_rmbg_weights_root);
                Ok((Some(root), None))
            }
            RmbgModel::Rmbg2 => {
                let root = args
                    .rmbg2_root
                    .clone()
                    .or_else(|| args.rmbg_root.clone())
                    .unwrap_or_else(resolve_rmbg2_weights_root);
                Ok((None, Some(root)))
            }
            RmbgModel::Both => {
                if let Some(shared) = args.rmbg_root.as_ref() {
                    let rmbg14 = args
                        .rmbg14_root
                        .clone()
                        .unwrap_or_else(|| shared.join("RMBG-1.4"));
                    let rmbg2 = args
                        .rmbg2_root
                        .clone()
                        .unwrap_or_else(|| shared.join("RMBG-2.0"));
                    return Ok((Some(rmbg14), Some(rmbg2)));
                }

                Ok((
                    Some(
                        args.rmbg14_root
                            .clone()
                            .unwrap_or_else(resolve_rmbg_weights_root),
                    ),
                    Some(
                        args.rmbg2_root
                            .clone()
                            .unwrap_or_else(resolve_rmbg2_weights_root),
                    ),
                ))
            }
        }
    }

    fn run_imports_with_backend<B>(
        use_f16: bool,
        overwrite: bool,
        rmbg14: Option<(&Path, &RmbgConfig)>,
        rmbg2_root: Option<&Path>,
        artifact_policy: ArtifactPolicy,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        B: burn::tensor::backend::Backend,
        B::Device: Default,
    {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        if let Some((weights, config)) = rmbg14 {
            import_if_needed(
                "RMBG-1.4",
                weights,
                use_f16,
                overwrite,
                artifact_policy,
                || import_rmbg_burnpack::<B>(&device, weights, config, use_f16),
            )?;
        }

        if let Some(root) = rmbg2_root {
            if !root.exists() {
                return Err(format!("missing RMBG-2.0 root at {}", root.display()).into());
            }
            let output = root.join(if use_f16 {
                "model_f16.bpk"
            } else {
                "model.bpk"
            });
            let precision = precision_label(use_f16);
            let output = if output.exists() && !overwrite {
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) burnpack already exists at {}, skipping.",
                    output.display()
                );
                output
            } else {
                ensure_parent_dir(&output)?;
                let saved = import_rmbg2_burnpack(root, use_f16)?;
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) burnpack saved to {}",
                    saved.display()
                );
                saved
            };
            if let Some(report) = apply_artifact_policy(&output, artifact_policy, overwrite)? {
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) parts manifest: {} ({} parts, {:.1} MiB)",
                    report.manifest_path.display(),
                    report.part_paths.len(),
                    report.total_bytes as f64 / (1024.0 * 1024.0),
                );
            }
            let removed_legacy = remove_legacy_shard_artifacts_for_burnpack(&output)?;
            if removed_legacy > 0 {
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) removed {removed_legacy} legacy shard artifact(s)"
                );
            }
        }

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
        if !weights_path.exists() {
            return Err(format!("missing {label} weights at {}", weights_path.display()).into());
        }

        let burnpack = burnpack_path(weights_path, use_f16);
        let precision = precision_label(use_f16);
        let output = if burnpack.exists() && !overwrite {
            println!(
                "[IMPORT] {label} ({precision}) burnpack already exists at {}, skipping.",
                burnpack.display()
            );
            burnpack
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
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

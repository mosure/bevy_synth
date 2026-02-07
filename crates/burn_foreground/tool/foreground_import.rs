#![recursion_limit = "256"]

#[cfg(target_arch = "wasm32")]
fn main() {
    eprintln!("foreground_import is unavailable on wasm32 targets");
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use burn::backend::NdArray;
    use clap::{Parser, ValueEnum};

    use burn_foreground::rmbg2::import::{import_rmbg2_burnpack, resolve_rmbg2_weights_root};
    use burn_foreground::rmbg14::{
        RmbgConfig,
        import::{import_rmbg_burnpack, load_rmbg_config, resolve_rmbg_weights_root},
    };

    type CpuBackend = NdArray<f32>;
    type GpuBackend = burn_wgpu::Wgpu;
    const F16_SUFFIX: &str = "_f16";

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
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        B: burn::tensor::backend::Backend,
        B::Device: Default,
    {
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        if let Some((weights, config)) = rmbg14 {
            import_if_needed("RMBG-1.4", weights, use_f16, overwrite, || {
                import_rmbg_burnpack::<B>(&device, weights, config, use_f16)
            })?;
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
            let precision = if use_f16 { "f16" } else { "f32" };
            if output.exists() && !overwrite {
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) burnpack already exists at {}, skipping.",
                    output.display()
                );
            } else {
                if let Some(parent) = output.parent() {
                    fs::create_dir_all(parent)?;
                }
                let saved = import_rmbg2_burnpack(root, use_f16)?;
                println!(
                    "[IMPORT] RMBG-2.0 ({precision}) burnpack saved to {}",
                    saved.display()
                );
            }
        }

        Ok(())
    }

    fn burnpack_path(path: &Path, use_f16: bool) -> PathBuf {
        let path = if path
            .extension()
            .map(|ext| ext.eq_ignore_ascii_case("bpk"))
            .unwrap_or(false)
        {
            path.to_path_buf()
        } else {
            path.with_extension("bpk")
        };

        if use_f16 {
            with_file_stem_suffix(&path, F16_SUFFIX)
        } else {
            path
        }
    }

    fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
        let Some(stem) = path.file_stem() else {
            return path.to_path_buf();
        };
        let stem = stem.to_string_lossy();
        if stem.ends_with(suffix) {
            return path.to_path_buf();
        }

        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        let mut file_name = format!("{stem}{suffix}");
        if !ext.is_empty() {
            file_name.push('.');
            file_name.push_str(ext);
        }
        path.with_file_name(file_name)
    }

    fn import_if_needed<F>(
        label: &str,
        weights_path: &Path,
        use_f16: bool,
        overwrite: bool,
        import_fn: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnOnce() -> Result<PathBuf, Box<dyn std::error::Error>>,
    {
        if !weights_path.exists() {
            return Err(format!("missing {label} weights at {}", weights_path.display()).into());
        }

        let burnpack = burnpack_path(weights_path, use_f16);
        let precision = if use_f16 { "f16" } else { "f32" };
        if burnpack.exists() && !overwrite {
            println!(
                "[IMPORT] {label} ({precision}) burnpack already exists at {}, skipping.",
                burnpack.display()
            );
            return Ok(());
        }

        if let Some(parent) = burnpack.parent() {
            fs::create_dir_all(parent)?;
        }

        let output = import_fn()?;
        println!(
            "[IMPORT] {label} ({precision}) burnpack saved to {}",
            output.display()
        );
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

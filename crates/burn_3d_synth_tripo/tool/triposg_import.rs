#![recursion_limit = "256"]

use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::backend::NdArray;
use clap::{Parser, ValueEnum};

use burn_3d_synth_bg_removal::model::{
    RmbgConfig,
    import::{import_rmbg_burnpack, load_rmbg_config, resolve_rmbg_weights_root},
};
use burn_3d_synth_tripo::model::triposg::{
    dit::{TripoSGDiTConfig, import::import_triposg_dit_burnpack},
    image_encoder::import::{import_triposg_dinov2_burnpack, resolve_triposg_weights_root},
    vae::{TripoSGVaeConfig, import::import_triposg_vae_burnpack},
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
    about = "Import TripoSG safetensors into Burnpack (.bpk) files",
    version
)]
struct Args {
    #[arg(long)]
    weights_root: Option<PathBuf>,

    #[arg(long)]
    rmbg_root: Option<PathBuf>,

    #[arg(long)]
    skip_rmbg: bool,

    #[arg(long)]
    overwrite: bool,

    #[arg(long, value_enum, default_value_t = Quantization::F32)]
    quantization: Quantization,
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

    let rmbg = if args.skip_rmbg {
        None
    } else {
        let rmbg_root = args
            .rmbg_root
            .clone()
            .unwrap_or_else(resolve_rmbg_weights_root);
        let rmbg_weights = rmbg_root.join("model.safetensors");
        let rmbg_config = load_rmbg_config(&rmbg_root)?;
        Some((rmbg_weights, rmbg_config))
    };

    if args.quantization.include_f32() {
        run_imports_with_backend::<CpuBackend>(
            false,
            args.overwrite,
            &vae_path,
            &dit_path,
            &dino_path,
            &vae_config,
            &dit_config,
            rmbg.as_ref()
                .map(|(weights, config)| (weights.as_path(), config)),
        )?;
    }
    if args.quantization.include_f16() {
        run_imports_with_backend::<GpuBackend>(
            true,
            args.overwrite,
            &vae_path,
            &dit_path,
            &dino_path,
            &vae_config,
            &dit_config,
            rmbg.as_ref()
                .map(|(weights, config)| (weights.as_path(), config)),
        )?;
    }

    Ok(())
}

fn run_imports_with_backend<B>(
    use_f16: bool,
    overwrite: bool,
    vae_path: &Path,
    dit_path: &Path,
    dino_path: &Path,
    vae_config: &TripoSGVaeConfig,
    dit_config: &TripoSGDiTConfig,
    rmbg: Option<(&Path, &RmbgConfig)>,
) -> Result<(), Box<dyn std::error::Error>>
where
    B: burn::tensor::backend::Backend,
    B::Device: Default,
{
    let device = <B as burn::tensor::backend::Backend>::Device::default();

    import_if_needed("VAE", vae_path, use_f16, overwrite, || {
        import_triposg_vae_burnpack::<B>(vae_config, &device, vae_path, use_f16)
    })?;

    import_if_needed("DiT", dit_path, use_f16, overwrite, || {
        import_triposg_dit_burnpack::<B>(dit_config, &device, dit_path, use_f16)
    })?;

    import_if_needed("DINOv2", dino_path, use_f16, overwrite, || {
        import_triposg_dinov2_burnpack::<B>(&device, dino_path, use_f16)
    })?;

    if let Some((rmbg_weights, rmbg_config)) = rmbg {
        import_if_needed("RMBG", rmbg_weights, use_f16, overwrite, || {
            import_rmbg_burnpack::<B>(&device, rmbg_weights, rmbg_config, use_f16)
        })?;
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

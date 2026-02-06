use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::backend::NdArray;
use clap::Parser;

use burn_3d_synth_bg_removal::model::import::{
    import_rmbg_burnpack, load_rmbg_config, resolve_rmbg_weights_root,
};
use burn_3d_synth_tripo::model::triposg::{
    dit::{TripoSGDiTConfig, import::import_triposg_dit_burnpack},
    image_encoder::import::{import_triposg_dinov2_burnpack, resolve_triposg_weights_root},
    vae::{TripoSGVaeConfig, import::import_triposg_vae_burnpack},
};

type Backend = NdArray<f32>;

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
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();

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

    import_if_needed("VAE", &vae_path, args.overwrite, || {
        import_triposg_vae_burnpack::<Backend>(&vae_config, &device, &vae_path)
    })?;

    import_if_needed("DiT", &dit_path, args.overwrite, || {
        import_triposg_dit_burnpack::<Backend>(&dit_config, &device, &dit_path)
    })?;

    import_if_needed("DINOv2", &dino_path, args.overwrite, || {
        import_triposg_dinov2_burnpack::<Backend>(&device, &dino_path)
    })?;

    if !args.skip_rmbg {
        let rmbg_root = args.rmbg_root.unwrap_or_else(resolve_rmbg_weights_root);
        let rmbg_weights = rmbg_root.join("model.safetensors");
        let rmbg_config = load_rmbg_config(&rmbg_root)?;
        import_if_needed("RMBG", &rmbg_weights, args.overwrite, || {
            import_rmbg_burnpack::<Backend>(&device, &rmbg_weights, &rmbg_config)
        })?;
    }

    Ok(())
}

fn burnpack_path(path: &Path) -> PathBuf {
    if path
        .extension()
        .map(|ext| ext.eq_ignore_ascii_case("bpk"))
        .unwrap_or(false)
    {
        path.to_path_buf()
    } else {
        path.with_extension("bpk")
    }
}

fn import_if_needed<F>(
    label: &str,
    weights_path: &Path,
    overwrite: bool,
    import_fn: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<PathBuf, Box<dyn std::error::Error>>,
{
    if !weights_path.exists() {
        return Err(format!("missing {label} weights at {}", weights_path.display()).into());
    }

    let burnpack = burnpack_path(weights_path);
    if burnpack.exists() && !overwrite {
        println!(
            "[IMPORT] {label} burnpack already exists at {}, skipping.",
            burnpack.display()
        );
        return Ok(());
    }

    if let Some(parent) = burnpack.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = import_fn()?;
    println!("[IMPORT] {label} burnpack saved to {}", output.display());
    Ok(())
}

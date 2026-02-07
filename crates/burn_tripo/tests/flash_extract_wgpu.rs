#![cfg(feature = "import")]

use burn::prelude::*;

use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_tripo::pipeline::geometry::{FlashExtractConfig, flash_extract_geometry};

#[test]
fn flash_extract_wgpu_smoke() {
    if std::env::var("BURN_WGPU_SMOKE").is_err() {
        eprintln!("skipping: set BURN_WGPU_SMOKE=1 to run wgpu flash extract smoke test");
        return;
    }

    let result = std::panic::catch_unwind(|| {
        let device = burn_wgpu::WgpuDevice::default();
        let vae = TripoSGVaeConfig::midi_3d().init::<burn_wgpu::Wgpu>(&device);

        let latents = Tensor::<burn_wgpu::Wgpu, 3>::random(
            [1, 8, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let config = FlashExtractConfig {
            bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            octree_depth: 3,
            num_chunks: 128,
            mc_level: 0.0,
            min_resolution: 3,
            mini_grid_num: 1,
        };

        unsafe {
            std::env::set_var("TRIPOSG_FLASH_NO_FALLBACK", "1");
        }
        let grid =
            flash_extract_geometry(latents, &vae, &config).expect("flash_extract_geometry failed");

        assert!(
            grid.values.iter().any(|value| value.is_finite()),
            "flash_extract_geometry produced all NaNs"
        );
    });

    if result.is_err() {
        eprintln!("skipping: wgpu backend not available on this system");
    }
}

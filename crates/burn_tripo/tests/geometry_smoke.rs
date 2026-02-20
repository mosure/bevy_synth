use burn::prelude::*;

use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_tripo::pipeline::geometry::{HierarchicalExtractConfig, hierarchical_extract_geometry};

#[test]
fn hierarchical_extract_smoke() {
    let device = Default::default();
    let vae_config = TripoSGVaeConfig::midi_3d();
    let vae = vae_config.init(&device);
    let latents = Tensor::<burn::backend::NdArray<f32>, 3>::zeros(
        [1, 8, vae_config.latent_channels as i32],
        &device,
    );

    let config = HierarchicalExtractConfig {
        bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        dense_octree_depth: 1,
        hierarchical_octree_depth: 2,
        chunk_size: 32,
        band_threshold: 1.0,
    };

    let grid = hierarchical_extract_geometry(&latents, &vae, &config)
        .expect("hierarchical extraction failed");
    assert_eq!(grid.size, [4, 4, 4]);
    assert_eq!(grid.values.len(), 4 * 4 * 4);
}

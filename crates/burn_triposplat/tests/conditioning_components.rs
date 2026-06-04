use burn::prelude::*;
use burn_dino::model::dinov3::DinoV3Config;
use burn_flux::Flux2VaeEncoderConfig;

type TestBackend = burn::backend::NdArray<f32>;

#[test]
fn dinov3_tiny_conditioning_shape_matches_triposplat_prefix_contract() {
    let device = Default::default();
    let config = DinoV3Config::tiny_for_tests(32, 16);
    let model = config.clone().init::<TestBackend>(&device);
    let image = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
    let features = model.forward(image);
    assert_eq!(
        features.dims(),
        [
            1,
            1 + config.num_register_tokens + (32 / config.patch_size).pow(2),
            config.hidden_size
        ]
    );
}

#[test]
fn flux2_vae_conditioning_shape_matches_triposplat_feature2_contract() {
    let device = Default::default();
    let model = Flux2VaeEncoderConfig::flux2().init::<TestBackend>(&device);
    let image = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
    let features = model.encode(image, true);
    assert_eq!(features.dims(), [1, 4, 128]);
}

#[cfg(feature = "import")]
#[test]
fn dinov3_import_remaps_hf_keys_to_burn_module_paths() {
    let mut remapper = burn_store::KeyRemapper::new();
    for &(from, to) in burn_dino::model::dinov3::import::dinov3_key_remap_rules() {
        remapper = remapper.add_pattern(from, to).unwrap();
    }
    let mut out = "encoder.layer.3.attention.q_proj.weight".to_string();
    for (pattern, replacement) in &remapper.patterns {
        if pattern.is_match(&out) {
            out = pattern.replace_all(&out, replacement.as_str()).to_string();
        }
    }
    assert_eq!(out, "blocks.3.attn.q_proj.weight");
}

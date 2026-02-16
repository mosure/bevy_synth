use std::path::PathBuf;

use burn::backend::NdArray;
use burn_tripo::model::triposg::vae::TripoSGVaeConfig;
use burn_tripo::model::triposg::vae::import::load_triposg_vae_decoder_from_burnpack_bytes;

#[test]
fn vae_f16_burnpack_loads_on_ndarray() {
    let path = PathBuf::from("www/assets/models/MIDI-3D/vae/diffusion_pytorch_model_f16.bpk");
    if !path.exists() {
        eprintln!("skipping: {} missing", path.display());
        return;
    }

    let bytes = std::fs::read(&path).expect("read f16 vae burnpack");
    let device = <NdArray<f32> as burn::prelude::Backend>::Device::default();
    let config = TripoSGVaeConfig::midi_3d();
    let result =
        load_triposg_vae_decoder_from_burnpack_bytes::<NdArray<f32>>(&config, &device, bytes);
    assert!(result.is_ok(), "f16 VAE load failed: {:?}", result.err());
}

pub mod flux2_vae;

#[cfg(feature = "import")]
pub use flux2_vae::import as flux2_import;
pub use flux2_vae::{
    Flux2Attention, Flux2Downsampler, Flux2Encoder, Flux2EncoderTrace, Flux2ResnetBlock,
    Flux2VaeEncodeTrace, Flux2VaeEncoder, Flux2VaeEncoderConfig, FrozenBatchNorm1d,
};

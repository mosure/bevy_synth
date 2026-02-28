#![recursion_limit = "256"]

pub mod config;
pub mod hook_diff;
mod hook_trace;
pub mod mesh;
pub mod paths;
pub mod pipeline;
pub mod preprocess;
pub mod sampler;
pub mod staged_pipeline;
pub mod trellis_config;

#[cfg(feature = "import")]
pub mod import;

#[cfg(feature = "runtime-model")]
pub(crate) mod blob_burnpack;

#[cfg(feature = "runtime-model")]
pub mod runtime_model;

pub use config::{TrellisQuality, TrellisQualitySettings};
pub use mesh::{
    Mesh, MeshMaterial, MeshPbrTextures, MeshTexture, load_obj_mesh, write_glb_mesh, write_obj_mesh,
};
pub use pipeline::{
    Trellis2Pipeline, Trellis2PipelineConfig, TrellisDevice, TrellisRunOptions, TrellisRuntimeError,
};

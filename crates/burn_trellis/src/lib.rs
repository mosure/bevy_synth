#![recursion_limit = "256"]
#![cfg_attr(
    feature = "runtime-model-wgpu",
    allow(
        clippy::collapsible_if,
        clippy::manual_div_ceil,
        clippy::manual_is_multiple_of,
        clippy::manual_memcpy,
        clippy::needless_return,
        clippy::single_range_in_vec_init,
        clippy::too_many_arguments,
        clippy::type_complexity,
        clippy::unnecessary_map_or,
        clippy::unnecessary_min_or_max,
        clippy::useless_conversion
    )
)]

pub mod config;
pub mod hook_diff;
mod hook_trace;
pub mod mesh;
pub mod paths;
pub mod pipeline;
pub mod preprocess;
pub mod sampler;
pub mod staged_pipeline;
pub(crate) mod time;
pub mod trellis_config;
pub mod virtual_fs;

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

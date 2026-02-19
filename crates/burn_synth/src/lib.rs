#![recursion_limit = "256"]

pub mod io;
pub mod mesh;
pub mod model_loader;
#[cfg(all(feature = "runtime", not(target_arch = "wasm32")))]
mod native_model_bootstrap;
pub mod pipeline;
#[cfg(feature = "runtime")]
pub mod progress;
#[cfg(feature = "runtime")]
pub mod runtime;
pub mod wasm;
#[cfg(all(target_arch = "wasm32", feature = "wasm-api"))]
pub mod wasm_api;
#[cfg(target_arch = "wasm32")]
pub mod wasm_loader;

#[cfg(any(feature = "runtime", feature = "wasm-api"))]
pub use io::mesh_to_glb_bytes;
#[cfg(feature = "runtime")]
pub use io::write_glb_mesh;
pub use io::{ImageSource, TextPrompt};
pub use mesh::{
    Mesh, MeshLike, MeshMaterial, MeshPbrTextures, MeshStats, MeshTexture, mesh_bounds, mesh_stats,
};
#[cfg(all(feature = "runtime", not(target_arch = "wasm32")))]
pub use native_model_bootstrap::set_bootstrap_status_callback;
pub use pipeline::{
    ForegroundModel, MeshOutput, ModelSelection, PipelineInput, PipelineOutput, SynthesisModel,
    sanitize_synthesis_models,
};
#[cfg(feature = "runtime")]
pub use progress::{
    ProgressCallback, ProgressVerbosity, RuntimeProgressEvent, RuntimeProgressObserver,
    default_log_progress_callback, log_progress_event,
};
#[cfg(feature = "runtime")]
pub use runtime::{
    DinoBackend, ForegroundOutput, ForegroundRequest, InferenceBackend,
    MeshOutput as RuntimeMeshOutput, MeshRequest, RuntimeConfig, RuntimeError, SynthRuntime,
    TrellisQuality,
};

#[cfg(feature = "triposg")]
pub use burn_tripo as triposg;

#[cfg(feature = "trellis")]
pub use burn_trellis as trellis;

#[cfg(feature = "bg-removal")]
pub use burn_foreground as bg_removal;

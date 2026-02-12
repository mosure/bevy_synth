#![recursion_limit = "256"]

pub mod io;
pub mod mesh;
pub mod pipeline;
#[cfg(feature = "runtime")]
pub mod progress;
#[cfg(feature = "runtime")]
pub mod runtime;

pub use io::{ImageSource, TextPrompt};
#[cfg(feature = "runtime")]
pub use io::{mesh_to_glb_bytes, write_glb_mesh};
pub use mesh::{
    Mesh, MeshLike, MeshMaterial, MeshPbrTextures, MeshStats, MeshTexture, mesh_bounds, mesh_stats,
};
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
};

#[cfg(feature = "triposg")]
pub use burn_tripo as triposg;

#[cfg(feature = "trellis")]
pub use burn_trellis as trellis;

#[cfg(feature = "bg-removal")]
pub use burn_foreground as bg_removal;

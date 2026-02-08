#![recursion_limit = "256"]

pub mod io;
pub mod mesh;
pub mod pipeline;
#[cfg(feature = "runtime")]
pub mod runtime;

pub use io::{ImageSource, TextPrompt};
pub use mesh::{Mesh, MeshLike, MeshStats, mesh_bounds, mesh_stats};
pub use pipeline::{
    ForegroundModel, MeshOutput, ModelSelection, PipelineInput, PipelineOutput, SynthesisModel,
    sanitize_synthesis_models,
};
#[cfg(feature = "runtime")]
pub use runtime::{
    ForegroundOutput, ForegroundRequest, InferenceBackend, MeshOutput as RuntimeMeshOutput,
    MeshRequest, RuntimeConfig, RuntimeError, SynthRuntime,
};

#[cfg(feature = "triposg")]
pub use burn_tripo as triposg;

#[cfg(feature = "trellis")]
pub use burn_trellis as trellis;

#[cfg(feature = "bg-removal")]
pub use burn_foreground as bg_removal;

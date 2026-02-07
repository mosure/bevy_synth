pub mod io;
pub mod mesh;
pub mod pipeline;

pub use io::{ImageSource, TextPrompt};
pub use mesh::{Mesh, MeshLike, MeshStats, mesh_bounds, mesh_stats};
pub use pipeline::{MeshOutput, PipelineInput, PipelineOutput};

#[cfg(feature = "triposg")]
pub use burn_tripo as triposg;

#[cfg(feature = "bg-removal")]
pub use burn_foreground as bg_removal;

#![recursion_limit = "256"]

mod assets;
mod cli;
mod feedback;
mod prelude;
mod protocol;
mod scene_layout;
mod server;
mod types;

pub use cli::run_from_args;
pub use protocol::SceneBuildFromImageArgs;
pub use server::run_stdio_server;
pub use types::{
    AssetOutputFormat, FeedbackRotationSelector, FeedbackThresholdProfile, ForegroundModel,
    InferenceBackend, LocateAnythingBackend, MeshOutputFormat, QualityPreset,
    SceneCanonicalPoseMode, SceneCompositionMode, SceneDepthPrecision, SceneDepthProvider,
    SceneLocatorProvider, ScenePoseFitMode, ServerArgs, ServerConfig, SynthesisModel,
    TrellisQuality,
};

pub fn run_scene_build_from_image(
    config: ServerConfig,
    args: SceneBuildFromImageArgs,
) -> Result<serde_json::Value, String> {
    let mut server = server::McpServer::new(config);
    server.call_scene_build_from_image(args)
}

#[cfg(test)]
pub(crate) use assets::*;
#[cfg(test)]
pub(crate) use feedback::*;
#[cfg(test)]
pub(crate) use protocol::*;
#[cfg(test)]
pub(crate) use server::McpServer;

#[cfg(test)]
mod tests;

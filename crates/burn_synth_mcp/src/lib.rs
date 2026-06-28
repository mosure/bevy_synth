#![recursion_limit = "256"]

mod assets;
mod canonical_pose;
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
    AssetOutputFormat, CubeClAutotuneCacheSetting, CubeClAutotuneLevelSetting,
    FeedbackRotationSelector, FeedbackThresholdProfile, ForegroundModel, InferenceBackend,
    LocateAnythingBackend, MeshOutputFormat, QualityPreset, SceneBuildExecutionKind,
    SceneBuildProgressEvent, SceneBuildProgressPhase, SceneCanonicalPoseMode, SceneCompositionMode,
    SceneDepthPrecision, SceneDepthProvider, SceneLocateAnythingPrecision, SceneLocatorProvider,
    ScenePoseFitMode, SceneSegmentationPrecision, SceneSegmentationProvider,
    SceneSegmentationQuantization, ServerArgs, ServerConfig, SynthesisModel, TrellisQuality,
};

pub fn run_scene_build_from_image(
    config: ServerConfig,
    args: SceneBuildFromImageArgs,
) -> Result<serde_json::Value, String> {
    run_scene_build_from_image_with_progress(config, args, |_| {})
}

pub fn run_scene_build_from_image_with_progress<F>(
    config: ServerConfig,
    args: SceneBuildFromImageArgs,
    mut progress: F,
) -> Result<serde_json::Value, String>
where
    F: FnMut(SceneBuildProgressEvent),
{
    let mut server = server::McpServer::new(config);
    server.call_scene_build_from_image_with_progress(args, &mut progress)
}

#[cfg(test)]
pub(crate) use assets::*;
#[cfg(test)]
pub(crate) use canonical_pose::*;
#[cfg(test)]
pub(crate) use feedback::*;
#[cfg(test)]
pub(crate) use protocol::*;
#[cfg(test)]
pub(crate) use server::McpServer;

#[cfg(test)]
mod tests;

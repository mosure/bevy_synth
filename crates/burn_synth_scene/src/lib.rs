mod bsn;
mod canonical_pose;
mod cli;
mod error;
mod layout;
mod object_images;
mod openai;
mod pipeline;
mod projection_fit;
mod types;

pub use bsn::{
    crop_scene_object, load_scene_asset_bindings, object_manifest_schema, parse_scene_bsn,
    resize_image_for_api, rotation_selection_schema, scene_bsn_file_to_mcp_command_envelope,
    scene_bsn_schema, scene_bsn_to_mcp_command_envelope, scene_plan_to_mcp_commands,
    write_json_file, write_metric,
};
pub use canonical_pose::{
    canonical_frame_for_asset, canonical_pose_evidence_for_asset,
    canonical_pose_evidence_for_assets, canonical_spawn_yaw_degrees, symmetry_for_descriptor,
};
pub use cli::{Cli, run_cli};
pub use error::{SceneError, SceneResult};
pub use layout::{
    GroundedSceneLayout, GroundedSceneLayoutConfig, GroundedScenePlacement, grounded_scene_bsn,
    grounded_scene_layout, grounded_scene_layout_for_manifest, grounded_scene_layout_with_evidence,
    manifest_grounding_evidence, manifest_with_grounding_evidence,
};
pub use object_images::{
    object_image_prompt, object_image_prompt_template, object_manifest_prompt, scene_bsn_prompt,
};
pub use openai::{OpenAiProviderConfig, OpenAiSceneProvider};
pub use pipeline::{
    SceneAiProvider, SceneBsnRequest, ScenePipeline, SceneReasoningRequest, SceneRotationSelection,
    SceneRotationSelectionRequest, SceneRotationSelectionResponse,
    object_image_candidate_rejections, select_object_image_candidates,
    select_object_image_candidates_with_exclusions,
};
pub use projection_fit::{
    ProjectionFitCandidateReport, ProjectionFitObjectReport, ProjectionFitReport,
};
pub use types::*;

#[cfg(test)]
mod tests;

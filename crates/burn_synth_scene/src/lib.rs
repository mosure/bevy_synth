mod bsn;
mod canonical_pose;
mod category_filter;
mod chair_types;
mod cli;
mod depth_sidecar;
mod error;
mod ground_calibration;
mod intrinsics;
mod layout;
mod object_images;
mod openai;
mod pipeline;
mod pose_fit;
mod pose_fit_prelude;
mod projection_fit;
mod types;
mod visible_surface_report;

pub use bsn::{
    crop_scene_object, load_scene_asset_bindings, object_manifest_schema, parse_scene_bsn,
    resize_image_for_api, rotation_selection_schema, scene_asset_declaration_for_bsn,
    scene_asset_source_for_bsn, scene_bsn_file_to_mcp_command_envelope, scene_bsn_schema,
    scene_bsn_to_mcp_command_envelope, scene_plan_to_mcp_commands, scene_quality_rubric_schema,
    write_json_file, write_metric,
};
pub use canonical_pose::{
    canonical_frame_for_asset, canonical_pose_evidence_for_asset,
    canonical_pose_evidence_for_assets, canonical_spawn_yaw_degrees, symmetry_for_descriptor,
};
pub use category_filter::{
    SceneCategoryFilterConfig, SceneCategoryFilterReport, SceneObjectCategory,
    ScenePipelineTomlConfig, ScenePipelineTomlGrounding, ScenePipelineTomlInstances,
    ScenePipelineTomlModels, ScenePipelineTomlOutput, ScenePipelineTomlScene,
    apply_scene_category_filter, canonical_scene_category, default_scene_category_allowlist,
    default_scene_category_denylist, infer_detection_category, infer_object_category,
    load_scene_pipeline_toml, parse_scene_category, scene_category_filter_prompt,
    scene_pipeline_toml_template,
};
pub use chair_types::{
    apply_chair_type_groups, chair_type_grouping_report, chair_type_grouping_schema,
    prepare_chair_type_grouping_request,
};
pub use cli::{Cli, run_cli};
pub use error::{SceneError, SceneResult};
pub use ground_calibration::{
    SceneGroundCalibrationReport, SceneGroundCalibrationRequest, SceneGroundCalibrationResponse,
    apply_ground_calibration_response, ground_calibration_schema,
    prepare_ground_calibration_request,
};
pub use intrinsics::source_camera_intrinsics_from_evidence;
pub use layout::{
    GroundedSceneLayout, GroundedSceneLayoutConfig, GroundedScenePlacement, grounded_scene_bsn,
    grounded_scene_layout, grounded_scene_layout_for_manifest, grounded_scene_layout_with_evidence,
    grounded_scene_layout_with_evidence_config, manifest_grounding_evidence,
    manifest_with_grounding_evidence,
};
pub use object_images::{
    object_image_prompt, object_image_prompt_template, object_manifest_prompt, scene_bsn_prompt,
};
pub use openai::{OpenAiProviderConfig, OpenAiSceneProvider};
pub use pipeline::{
    SceneAiProvider, SceneBsnRequest, SceneChairTypeCrop, SceneChairTypeGroup,
    SceneChairTypeGroupingRequest, SceneChairTypeGroupingResponse, ScenePipeline,
    SceneQualityRubricIssue, SceneQualityRubricRequest, SceneQualityRubricResponse,
    SceneReasoningRequest, SceneRotationSelection, SceneRotationSelectionRequest,
    SceneRotationSelectionResponse, object_image_candidate_rejections,
    select_object_image_candidates, select_object_image_candidates_with_exclusions,
};
pub use pose_fit::{
    SceneFinalYawRefinementConfig, SceneObjectPoseRefinementConfig, ScenePoseFitObjectFilter,
    SceneRotationFitConfig, SceneRotationFitOutcome, SceneVisibleSurfacePoseFitConfig,
    apply_scene_final_yaw_refinement, apply_scene_object_pose_refinement, apply_scene_rotation_fit,
    apply_scene_visible_surface_pose_fit,
};
pub use projection_fit::{
    ProjectionFitCameraReport, ProjectionFitCandidateReport, ProjectionFitObjectReport,
    ProjectionFitReport, ProjectionFitVisibleSurfaceReport,
};
pub use types::*;
pub use visible_surface_report::write_visible_surface_fit_artifacts;

#[cfg(test)]
mod tests;

pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::env;
pub(crate) use std::fs;
pub(crate) use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::process::{Child, Command, Stdio};
pub(crate) use std::sync::atomic::Ordering;
pub(crate) use std::sync::{Arc, Mutex};
pub(crate) use std::thread;
pub(crate) use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub(crate) use bevy_synth_runtime::cache::{
    CachedAssetAabb, CachedCameraState, CachedMeshMetadata, CachedSceneMetrics, CachedScenePayload,
    CachedWorldItem, MeshCache,
};
pub(crate) use bevy_synth_runtime::{
    SynthMesh as CachedSynthMesh, SynthMeshMaterial as CachedSynthMeshMaterial,
    SynthMeshPbrTextures as CachedSynthMeshPbrTextures, SynthMeshTexture as CachedSynthMeshTexture,
    TripoMesh as CachedTripoMesh,
};
pub(crate) use burn_segmentation::BinaryMask;
pub(crate) use burn_synth::{
    AssetBatchItem, AssetBatchRequest, ForegroundRequest, ImageSource, Mesh, ProgressVerbosity,
    RuntimeBatchPolicy, RuntimeProgressEvent, RuntimeProgressObserver, SynthRuntime,
    SynthesisAsset, mesh_quality_failures, mesh_quality_metrics, write_glb_mesh,
};
pub(crate) use burn_synth_grounding::{
    DepthProGroundingConfig, DepthProGroundingReport, LocateAnythingGroundingConfig,
    LocateAnythingGroundingReport, SceneGroundingRuntime, SegmentationGroundingConfig,
    SegmentationGroundingReport, SegmentationModelKind, SegmentationRuntimeBackend,
};
pub(crate) use burn_synth_render::normal::{
    BinaryMaskView, DepthMapView, DepthNormalIntrinsics, MeshNormalInput, RenderAabb,
    SourceDepthNormalInput, normal_map_similarity,
    write_candidate_mesh_normal_render as write_render_candidate_mesh_normal,
    write_source_depth_normal_evidence as write_render_source_depth_normal_evidence,
};
pub(crate) use burn_synth_scene::{
    CanonicalPoseCalibrationReport, CanonicalPoseCandidate, CanonicalPoseSelection,
    DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE, GptDelegationRole, GroundedSceneLayout,
    GroundedSceneLayoutConfig, GroundedScenePlacement, GroundingContractEntry,
    GroundingContractReport, GroundingVerificationStatus, ObjectGroundingEvidence,
    ObjectImageGenerationPolicy, ObjectImageRequest, OpenAiProviderConfig, OpenAiSceneProvider,
    SceneAiProvider, SceneAssetAabb, SceneAssetBinding, SceneAssetFrame, SceneAssetFrameSource,
    SceneAssetSymmetry, SceneBsnRequest, SceneBuildConfig, SceneDecisionLog, SceneDecisionLogEntry,
    SceneFinalYawRefinementConfig, SceneFinalYawRefinementMode, SceneGroundCalibrationReport,
    SceneGroundingEvidence, SceneObjectManifest, SceneObjectPoseRefinementConfig,
    SceneObjectPoseRefinementMode, SceneObjectPoseRefinementSet, ScenePipeline, ScenePlan,
    ScenePoseFitMode, ScenePoseFitObjectFilter, SceneQualityProfile, SceneQualityRubricRequest,
    SceneQualityRubricResponse, SceneReasoningRequest, SceneResult, SceneRotationFitConfig,
    SceneRotationFitMode, SceneRotationFitOutcome, SceneRotationSelectionRequest,
    SceneRotationSelectionResponse, SceneScalePolicy, SceneVisibleSurfacePoseFitConfig,
    apply_chair_type_groups, apply_ground_calibration_response, apply_scene_final_yaw_refinement,
    apply_scene_object_pose_refinement, apply_scene_rotation_fit,
    apply_scene_visible_surface_pose_fit, canonical_pose_evidence_for_assets,
    chair_type_grouping_report, grounded_scene_layout, grounded_scene_layout_for_manifest,
    grounded_scene_layout_with_evidence_config, manifest_grounding_evidence,
    manifest_with_grounding_evidence, parse_scene_bsn, scene_asset_declaration_for_bsn,
    scene_plan_to_mcp_commands, select_object_image_candidates_with_exclusions,
    symmetry_for_descriptor, write_json_file, write_visible_surface_fit_artifacts,
};
pub(crate) use serde::{Deserialize, de::DeserializeOwned};
pub(crate) use serde_json::{Value, json};

pub(crate) use crate::assets::*;
pub(crate) use crate::canonical_pose::*;
pub(crate) use crate::feedback::*;
pub(crate) use crate::protocol::*;
pub(crate) use crate::scene_depth_sidecar::*;
pub(crate) use crate::scene_layout::{
    SceneComposeArgs, SceneComposePlan, SceneValidateArgs, compose_scene_layout,
    validate_scene_layout,
};
pub(crate) use crate::scene_pipeline_strategy::*;
pub(crate) use crate::types::*;

use crate::types::*;
use burn_synth_scene::SceneScalePolicy;
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
pub(crate) enum ScenePlacementEntryPoint {
    SceneBuild,
    SceneGround,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScenePlacementPipelineSelection {
    pub(crate) entry_point: ScenePlacementEntryPoint,
    pub(crate) lift_assets: bool,
    pub(crate) composition_mode: SceneCompositionMode,
    pub(crate) pose_fit: ScenePoseFitMode,
    pub(crate) canonical_pose: SceneCanonicalPoseMode,
    pub(crate) scale_policy: SceneScalePolicy,
    pub(crate) ground_calibration: SceneGroundCalibrationMode,
    pub(crate) instance_generation: SceneInstanceGenerationMode,
    pub(crate) depth_provider: SceneDepthProvider,
    pub(crate) locator: SceneLocatorProvider,
    pub(crate) segmentation_provider: SceneSegmentationProvider,
    pub(crate) feedback: bool,
    pub(crate) feedback_iters: usize,
    pub(crate) feedback_rotation_selector: FeedbackRotationSelector,
    pub(crate) feedback_rubric_scorer: FeedbackRubricScorer,
    pub(crate) rotation_fit: SceneRotationFitMode,
    pub(crate) table_pose_refinement: SceneTablePoseRefinementMode,
    pub(crate) max_pose_candidates: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ScenePlacementPipelinePlan {
    pub(crate) schema_version: u32,
    pub(crate) objective: &'static str,
    pub(crate) quality_profile: &'static str,
    pub(crate) entry_point: &'static str,
    pub(crate) stages: Vec<ScenePlacementStageSpec>,
    pub(crate) evidence_contracts: Vec<ScenePlacementEvidenceContract>,
    pub(crate) ablation_axes: Vec<ScenePlacementAblationAxis>,
    pub(crate) active_pose_optimizer: &'static str,
    pub(crate) warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ScenePlacementStageSpec {
    pub(crate) stage: &'static str,
    pub(crate) role: &'static str,
    pub(crate) method: String,
    pub(crate) enabled: bool,
    pub(crate) status: &'static str,
    pub(crate) mutual_exclusion_group: &'static str,
    pub(crate) evidence_inputs: Vec<&'static str>,
    pub(crate) outputs: Vec<&'static str>,
    pub(crate) objective: &'static str,
    pub(crate) gpt_role: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ScenePlacementEvidenceContract {
    pub(crate) evidence: &'static str,
    pub(crate) producer_stage: &'static str,
    pub(crate) consumers: Vec<&'static str>,
    pub(crate) required_for_best_quality: bool,
    pub(crate) status: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ScenePlacementAblationAxis {
    pub(crate) axis: &'static str,
    pub(crate) selected: String,
    pub(crate) options: Vec<&'static str>,
    pub(crate) mutual_exclusion_group: &'static str,
    pub(crate) expected_quality_impact: &'static str,
}

impl ScenePlacementPipelinePlan {
    pub(crate) fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|err| {
            json!({
                "schema_version": self.schema_version,
                "status": "serialization_error",
                "error": err.to_string(),
            })
        })
    }
}

pub(crate) fn scene_placement_pipeline_plan(
    selection: ScenePlacementPipelineSelection,
) -> ScenePlacementPipelinePlan {
    let cv_grounded = selection.composition_mode == SceneCompositionMode::CvGrounded;
    let has_depth = cv_grounded && selection.depth_provider == SceneDepthProvider::DepthPro;
    let has_gpt_ground_calibration =
        cv_grounded && selection.ground_calibration == SceneGroundCalibrationMode::Gpt;
    let has_locator = cv_grounded && selection.locator == SceneLocatorProvider::LocateAnything;
    let has_sam = cv_grounded
        && matches!(
            selection.segmentation_provider,
            SceneSegmentationProvider::Sam2 | SceneSegmentationProvider::Sam3
        );
    let has_any_mask =
        cv_grounded && selection.segmentation_provider != SceneSegmentationProvider::None;
    let asset_pose_enabled = selection.lift_assets && cv_grounded;
    let rendered_silhouette =
        asset_pose_enabled && selection.pose_fit == ScenePoseFitMode::RenderedSilhouette;
    let dense_pose_enabled = rendered_silhouette && has_depth && has_any_mask;
    let table_refinement_enabled =
        dense_pose_enabled && selection.table_pose_refinement.geometry_enabled();
    let feedback_enabled =
        selection.lift_assets && selection.feedback && selection.feedback_iters > 0;

    let mut warnings = Vec::new();
    if !selection.lift_assets {
        warnings
            .push("asset lifting is disabled; mesh-to-image pose fitting cannot run".to_string());
    }
    if selection.max_pose_candidates < 16 {
        warnings.push(
            "max_pose_candidates is below 16; deterministic yaw/scale search may miss viable transforms"
                .to_string(),
        );
    }
    if selection.feedback && selection.feedback_iters == 0 {
        warnings.push(
            "feedback was requested with zero iterations; render-capture validation is disabled"
                .to_string(),
        );
    }
    if !cv_grounded {
        warnings.push(
            "composition mode is heuristic; LocateAnything/SAM/DepthPro evidence is not authoritative"
                .to_string(),
        );
    }
    if cv_grounded && !has_locator {
        warnings.push(
            "LocateAnything is not selected; object cardinality/discretization falls back to manifest boxes"
                .to_string(),
        );
    }
    if cv_grounded && !has_sam {
        warnings.push(
            "SAM masks are not selected; depth point selection is degraded to bbox or disabled masks"
                .to_string(),
        );
    }
    if cv_grounded && !has_depth {
        warnings.push(
            "DepthPro is not selected; metric depth sidecar and unprojected point evidence are unavailable"
                .to_string(),
        );
    }
    if selection.scale_policy != SceneScalePolicy::AssetPreserving {
        warnings.push(
            "scale policy allows anisotropy; this can distort lifted tables/chairs and should be an explicit ablation"
                .to_string(),
        );
    }
    if asset_pose_enabled && selection.pose_fit == ScenePoseFitMode::ProjectedAabb {
        warnings.push(
            "projected-aabb pose fit is selected; dense mask/depth visible-surface fitting is disabled"
                .to_string(),
        );
    }
    if selection.rotation_fit == SceneRotationFitMode::GptRefine {
        warnings.push(
            "GPT rotation refinement is selected; it must remain bounded candidate selection, not a source of geometry truth"
                .to_string(),
        );
    }
    if feedback_enabled {
        warnings.push(
            "render-capture feedback is enabled; this is an explicit validation/refinement ablation, not the default scene composition path"
                .to_string(),
        );
    }
    if selection.feedback_rotation_selector == FeedbackRotationSelector::Openai {
        warnings.push(
            "OpenAI feedback rotation selector is selected; use only as bounded candidate selection after deterministic geometry scoring"
                .to_string(),
        );
    }
    if selection.feedback_rubric_scorer == FeedbackRubricScorer::Openai {
        warnings.push(
            "OpenAI scene rubric is selected; treat it as diagnostic scoring, not geometric ground truth"
                .to_string(),
        );
    }

    let quality_profile = if warnings.is_empty()
        && dense_pose_enabled
        && selection.scale_policy == SceneScalePolicy::AssetPreserving
        && selection.feedback_rotation_selector == FeedbackRotationSelector::Deterministic
        && selection.feedback_rubric_scorer == FeedbackRubricScorer::Off
        && selection.rotation_fit == SceneRotationFitMode::Off
        && selection.table_pose_refinement == SceneTablePoseRefinementMode::GatedGpt
    {
        "bare_bones_geometric"
    } else if dense_pose_enabled {
        "experimental_or_degraded"
    } else {
        "fallback"
    };

    let mut stages = Vec::new();
    stages.push(stage(StageTemplate {
        stage: "object_discretization",
        role: "evidence",
        method: if has_locator {
            "locate_anything_burn_native"
        } else if cv_grounded {
            "manifest_fallback"
        } else {
            "disabled"
        },
        enabled: cv_grounded,
        status: if has_locator { "active" } else { "fallback" },
        mutual_exclusion_group: "object_locator",
        evidence_inputs: vec!["source_image"],
        outputs: vec!["object_bboxes", "object_points", "object_cardinality"],
        objective: "Detect object instances; LocateAnything boxes are the semantic/cardinality source of truth when active.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "mask_selection",
        role: "evidence",
        method: match selection.segmentation_provider {
            SceneSegmentationProvider::Sam2 => "sam2",
            SceneSegmentationProvider::Sam3 => "sam3",
            SceneSegmentationProvider::BboxPrompt => "bbox_prompt_fallback",
            SceneSegmentationProvider::None => "disabled",
        },
        enabled: cv_grounded && has_any_mask,
        status: if has_sam {
            "active"
        } else if has_any_mask {
            "fallback"
        } else {
            "disabled"
        },
        mutual_exclusion_group: "mask_provider",
        evidence_inputs: vec!["source_image", "object_bboxes"],
        outputs: vec!["object_masks", "visible_pixel_selection"],
        objective: "Select visible source-image pixels per located object before depth unprojection.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "metric_depth",
        role: "evidence",
        method: if has_depth {
            "depth_pro_f32le_sidecar"
        } else {
            "disabled"
        },
        enabled: has_depth,
        status: if has_depth { "active" } else { "disabled" },
        mutual_exclusion_group: "depth_provider",
        evidence_inputs: vec!["source_image", "object_masks"],
        outputs: vec![
            "depth_map_f32le",
            "camera_intrinsics",
            "floor_plane",
            "object_depth_stats",
        ],
        objective: "Produce metric Z-depth, camera intrinsics, and floor calibration used as geometric ground truth.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "ground_calibration",
        role: "evidence_refinement",
        method: match selection.ground_calibration {
            SceneGroundCalibrationMode::DepthHeuristic => "depth_floor_heuristic",
            SceneGroundCalibrationMode::Gpt => "gpt_camera_floor",
        },
        enabled: cv_grounded,
        status: if has_gpt_ground_calibration {
            "active"
        } else if has_depth {
            "heuristic"
        } else {
            "fallback"
        },
        mutual_exclusion_group: "ground_calibration",
        evidence_inputs: vec!["source_image", "metric_depth", "object_bboxes"],
        outputs: vec!["estimated_camera", "estimated_floor_plane"],
        objective: "Choose the camera height/FOV/floor relation consumed by ray-floor object placement.",
        gpt_role: if has_gpt_ground_calibration {
            "hypothesis"
        } else {
            "none"
        },
    }));
    stages.push(stage(StageTemplate {
        stage: "object_instance_generation",
        role: "asset_input_planning",
        method: match selection.instance_generation {
            SceneInstanceGenerationMode::CategoryRepresentative => "category_representative",
            SceneInstanceGenerationMode::FineGrainedTypes => "gpt_fine_grained_types",
        },
        enabled: selection.lift_assets,
        status: if selection.instance_generation == SceneInstanceGenerationMode::FineGrainedTypes {
            "active"
        } else {
            "default"
        },
        mutual_exclusion_group: "instance_generation",
        evidence_inputs: vec!["object_bboxes", "source_object_crops"],
        outputs: vec!["reusable_object_groups"],
        objective: "Control whether repeated same-category instances share one generated asset or split into visual subtypes before expensive image-to-3D lifting.",
        gpt_role: if selection.instance_generation == SceneInstanceGenerationMode::FineGrainedTypes {
            "bounded_candidate_selection"
        } else {
            "none"
        },
    }));
    stages.push(stage(StageTemplate {
        stage: "object_image_synthesis",
        role: "asset_input",
        method: if selection.lift_assets {
            "gpt_image_2_crop_plus_reference_prompt"
        } else {
            "existing_assets"
        },
        enabled: selection.lift_assets,
        status: if selection.lift_assets { "active" } else { "skipped" },
        mutual_exclusion_group: "asset_source",
        evidence_inputs: vec!["source_object_crops", "object_prompt", "reference_object_image"],
        outputs: vec!["isolated_object_images"],
        objective: "Generate clean isolated reconstruction inputs only; GPT does not decide scene transforms.",
        gpt_role: if selection.lift_assets {
            "image_synthesis"
        } else {
            "none"
        },
    }));
    stages.push(stage(StageTemplate {
        stage: "image_to_3d_lifting",
        role: "asset_reconstruction",
        method: if selection.lift_assets {
            "trellis2_pbr"
        } else {
            "existing_asset_bindings"
        },
        enabled: selection.lift_assets,
        status: if selection.lift_assets { "active" } else { "skipped" },
        mutual_exclusion_group: "asset_source",
        evidence_inputs: vec!["isolated_object_images"],
        outputs: vec!["glb_assets", "asset_aabbs"],
        objective: "Lift each isolated object image into one reusable 3D asset plus local AABB metadata.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "initial_layout",
        role: "initialization",
        method: if cv_grounded {
            "camera_ray_floor_grounded"
        } else {
            "heuristic_layout"
        },
        enabled: true,
        status: if cv_grounded { "active" } else { "fallback" },
        mutual_exclusion_group: "layout_initializer",
        evidence_inputs: vec!["object_bboxes", "camera_intrinsics", "floor_plane", "asset_aabbs"],
        outputs: vec!["grounded_layout", "projection_fit_report"],
        objective: "Initialize object translation from source-camera rays, visible-surface depth, floor contact, and asset AABBs; keep scale uniform.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "canonical_asset_pose",
        role: "asset_frame",
        method: match selection.canonical_pose {
            SceneCanonicalPoseMode::Off => "disabled",
            SceneCanonicalPoseMode::Heuristic => "heuristic",
            SceneCanonicalPoseMode::RenderSweep => "render_sweep",
            SceneCanonicalPoseMode::Openai => "openai_candidate_selection",
            SceneCanonicalPoseMode::Auto => "auto",
        },
        enabled: asset_pose_enabled && selection.canonical_pose != SceneCanonicalPoseMode::Off,
        status: if selection.canonical_pose == SceneCanonicalPoseMode::RenderSweep {
            "active"
        } else if selection.canonical_pose == SceneCanonicalPoseMode::Off {
            "disabled"
        } else {
            "fallback_or_experimental"
        },
        mutual_exclusion_group: "canonical_pose",
        evidence_inputs: vec!["source_object_crops", "generated_asset_render_sweeps"],
        outputs: vec!["asset_frame_yaw_offsets"],
        objective: "Normalize each lifted asset frame before scene-space yaw optimization using rendered yaw candidates against source/generated object evidence.",
        gpt_role: if selection.canonical_pose == SceneCanonicalPoseMode::Openai {
            "bounded_candidate_selection"
        } else {
            "none"
        },
    }));
    stages.push(stage(StageTemplate {
        stage: "pose_optimizer",
        role: "optimization",
        method: if dense_pose_enabled {
            "visible_surface_dense_depth_search"
        } else if asset_pose_enabled && selection.pose_fit == ScenePoseFitMode::RenderedSilhouette {
            "visible_surface_summary_depth_search"
        } else if asset_pose_enabled {
            "projected_aabb"
        } else {
            "disabled"
        },
        enabled: asset_pose_enabled,
        status: if dense_pose_enabled {
            "active"
        } else if asset_pose_enabled {
            "fallback"
        } else {
            "disabled"
        },
        mutual_exclusion_group: "pose_optimizer",
        evidence_inputs: vec![
            "object_masks",
            "depth_map_f32le",
            "camera_intrinsics",
            "asset_meshes",
        ],
        outputs: vec!["pose_fit_candidates", "updated_scene_commands"],
        objective: "Search bounded camera-ray/floor-constrained X/Z/yaw/uniform-scale candidates and score projected mesh visible surfaces against source masks/depth.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "continuous_refinement",
        role: "optimization_refinement",
        method: if dense_pose_enabled {
            "burn_soft_point_surface"
        } else {
            "disabled"
        },
        enabled: dense_pose_enabled,
        status: if dense_pose_enabled {
            "active"
        } else {
            "disabled"
        },
        mutual_exclusion_group: "continuous_refinement",
        evidence_inputs: vec![
            "pose_fit_best_candidate",
            "object_depth_crop",
            "object_mask_crop",
            "asset_mesh_points",
        ],
        outputs: vec!["dense_soft_surface_candidate"],
        objective: "Refine the best deterministic candidate with a bounded differentiable soft point-surface loss, then re-score through deterministic gates.",
        gpt_role: "none",
    }));
    stages.push(stage(StageTemplate {
        stage: "table_pose_refinement",
        role: "optimization_specialization",
        method: match selection.table_pose_refinement {
            SceneTablePoseRefinementMode::Off => "disabled",
            SceneTablePoseRefinementMode::Geometry => "table_only_visible_surface_geometry",
            SceneTablePoseRefinementMode::GatedGpt => "table_only_geometry_with_gpt_gate",
            SceneTablePoseRefinementMode::AlwaysGpt => "table_only_geometry_with_required_gpt_gate",
        },
        enabled: table_refinement_enabled,
        status: if table_refinement_enabled {
            "active"
        } else if selection.table_pose_refinement == SceneTablePoseRefinementMode::Off {
            "disabled"
        } else {
            "waiting_for_dense_pose_inputs"
        },
        mutual_exclusion_group: "table_pose_refinement",
        evidence_inputs: vec![
            "pose_fit_candidates",
            "object_masks",
            "depth_map_f32le",
            "asset_meshes",
        ],
        outputs: vec![
            "table_pose_candidates",
            "updated_table_scene_commands",
            "gpt_required_table_fit_flags",
        ],
        objective: "Apply a stricter table-only X/Z/yaw/uniform-scale refinement after the generic pose solve; large table distortions stay disallowed.",
        gpt_role: if selection.table_pose_refinement.gpt_allowed() {
            "bounded_candidate_selection_after_failed_or_ambiguous_geometry"
        } else {
            "none"
        },
    }));
    stages.push(stage(StageTemplate {
        stage: "render_capture_feedback",
        role: "validation",
        method: if feedback_enabled {
            "bounded_feedback"
        } else {
            "disabled"
        },
        enabled: feedback_enabled,
        status: if feedback_enabled {
            "active"
        } else {
            "disabled"
        },
        mutual_exclusion_group: "feedback_validator",
        evidence_inputs: vec![
            "source_image",
            "rendered_scene",
            "scene_commands",
            "object_projection_metrics",
        ],
        outputs: vec![
            "accepted_scene_candidate",
            "feedback_metrics",
            "iteration_screenshots",
        ],
        objective: "Optional validation/refinement after the deterministic geometric solve; disabled in the default bare-bones flow.",
        gpt_role: if selection.feedback_rubric_scorer == FeedbackRubricScorer::Openai
            || selection.feedback_rotation_selector == FeedbackRotationSelector::Openai
        {
            "bounded_diagnosis_or_candidate_selection"
        } else {
            "none"
        },
    }));

    let evidence_contracts = vec![
        evidence_contract(
            "object instances and bboxes",
            "object_discretization",
            vec!["mask_selection", "initial_layout", "pose_optimizer"],
            true,
            if has_locator { "verified" } else { "fallback" },
        ),
        evidence_contract(
            "object visible masks",
            "mask_selection",
            vec![
                "metric_depth",
                "pose_optimizer",
                "continuous_refinement",
                "table_pose_refinement",
            ],
            true,
            if has_sam {
                "verified"
            } else if has_any_mask {
                "fallback"
            } else {
                "absent"
            },
        ),
        evidence_contract(
            "metric depth sidecar",
            "metric_depth",
            vec![
                "initial_layout",
                "pose_optimizer",
                "continuous_refinement",
                "table_pose_refinement",
            ],
            true,
            if has_depth { "verified" } else { "absent" },
        ),
        evidence_contract(
            "camera floor calibration",
            "ground_calibration",
            vec!["initial_layout", "pose_optimizer"],
            true,
            if has_gpt_ground_calibration {
                "hypothesis"
            } else if has_depth {
                "heuristic"
            } else {
                "fallback"
            },
        ),
        evidence_contract(
            "asset canonical frame",
            "canonical_asset_pose",
            vec![
                "pose_optimizer",
                "table_pose_refinement",
                "render_capture_feedback",
            ],
            false,
            if selection.canonical_pose == SceneCanonicalPoseMode::RenderSweep {
                "verified"
            } else if selection.canonical_pose == SceneCanonicalPoseMode::Off {
                "absent"
            } else {
                "fallback_or_experimental"
            },
        ),
    ];

    let ablation_axes = vec![
        ablation_axis(
            "object_discretization",
            if has_locator {
                "locate_anything"
            } else {
                "manifest"
            },
            vec!["manifest", "locate_anything"],
            "object_locator",
            "High: object count/cardinality errors propagate through every later stage.",
        ),
        ablation_axis(
            "mask_provider",
            match selection.segmentation_provider {
                SceneSegmentationProvider::None => "none",
                SceneSegmentationProvider::BboxPrompt => "bbox_prompt",
                SceneSegmentationProvider::Sam2 => "sam2",
                SceneSegmentationProvider::Sam3 => "sam3",
            },
            vec!["none", "bbox_prompt", "sam2", "sam3"],
            "mask_provider",
            "High: masks select which depth pixels become object geometry evidence.",
        ),
        ablation_axis(
            "metric_depth_provider",
            if has_depth { "depth_pro" } else { "none" },
            vec!["none", "depth_pro"],
            "depth_provider",
            "High: metric depth is the geometric source of truth for scale/position.",
        ),
        ablation_axis(
            "ground_calibration",
            match selection.ground_calibration {
                SceneGroundCalibrationMode::DepthHeuristic => "depth_heuristic",
                SceneGroundCalibrationMode::Gpt => "gpt",
            },
            vec!["depth_heuristic", "gpt"],
            "ground_calibration",
            "High: camera height/FOV/floor errors mirror or translate every object placement.",
        ),
        ablation_axis(
            "instance_generation",
            match selection.instance_generation {
                SceneInstanceGenerationMode::CategoryRepresentative => "category_representative",
                SceneInstanceGenerationMode::FineGrainedTypes => "fine_grained_types",
            },
            vec!["category_representative", "fine_grained_types"],
            "instance_generation",
            "Medium/high: fine-grained same-category assets can improve semantics but multiplies image generation and lifting cost.",
        ),
        ablation_axis(
            "asset_source",
            if selection.lift_assets {
                "gpt_image_2_trellis2"
            } else {
                "existing_asset_bindings"
            },
            vec!["existing_asset_bindings", "gpt_image_2_trellis2"],
            "asset_source",
            "Medium: composition can be rerun against existing assets without regenerating images or meshes.",
        ),
        ablation_axis(
            "pose_optimizer",
            if selection.pose_fit == ScenePoseFitMode::RenderedSilhouette {
                "rendered_silhouette_dense_depth"
            } else {
                "projected_aabb"
            },
            vec!["projected_aabb", "rendered_silhouette_dense_depth"],
            "pose_optimizer",
            "High: this is the direct object transform solver.",
        ),
        ablation_axis(
            "table_pose_refinement",
            match selection.table_pose_refinement {
                SceneTablePoseRefinementMode::Off => "off",
                SceneTablePoseRefinementMode::Geometry => "geometry",
                SceneTablePoseRefinementMode::GatedGpt => "gated_gpt",
                SceneTablePoseRefinementMode::AlwaysGpt => "always_gpt",
            },
            vec!["off", "geometry", "gated_gpt", "always_gpt"],
            "table_pose_refinement",
            "Medium/high: table-only retry can fix elongated table rotation/position without re-running expensive all-object feedback.",
        ),
        ablation_axis(
            "scale_policy",
            match selection.scale_policy {
                SceneScalePolicy::AssetPreserving => "asset_preserving",
                SceneScalePolicy::BoundedAnisotropic => "bounded_anisotropic",
                SceneScalePolicy::FreeAnisotropic => "free_anisotropic",
            },
            vec![
                "asset_preserving",
                "bounded_anisotropic",
                "free_anisotropic",
            ],
            "scale_policy",
            "Medium/high: non-uniform scale can improve bbox fit but risks distorted geometry.",
        ),
    ];

    ScenePlacementPipelinePlan {
        schema_version: 1,
        objective: "bare-bones source-camera geometric composition: LocateAnything boxes plus SAM-selected DepthPro point groups constrain projected lifted mesh visible surfaces",
        quality_profile,
        entry_point: match selection.entry_point {
            ScenePlacementEntryPoint::SceneBuild => "scene_build",
            ScenePlacementEntryPoint::SceneGround => "scene_ground",
        },
        stages,
        evidence_contracts,
        ablation_axes,
        active_pose_optimizer: if dense_pose_enabled {
            if table_refinement_enabled {
                "visible_surface_dense_depth_search_plus_soft_point_refinement_plus_table_refinement"
            } else {
                "visible_surface_dense_depth_search_plus_soft_point_refinement"
            }
        } else if asset_pose_enabled && selection.pose_fit == ScenePoseFitMode::RenderedSilhouette {
            "visible_surface_summary_depth_search"
        } else if asset_pose_enabled {
            "projected_aabb"
        } else {
            "disabled"
        },
        warnings,
    }
}

struct StageTemplate {
    stage: &'static str,
    role: &'static str,
    method: &'static str,
    enabled: bool,
    status: &'static str,
    mutual_exclusion_group: &'static str,
    evidence_inputs: Vec<&'static str>,
    outputs: Vec<&'static str>,
    objective: &'static str,
    gpt_role: &'static str,
}

fn stage(template: StageTemplate) -> ScenePlacementStageSpec {
    ScenePlacementStageSpec {
        stage: template.stage,
        role: template.role,
        method: template.method.to_string(),
        enabled: template.enabled,
        status: template.status,
        mutual_exclusion_group: template.mutual_exclusion_group,
        evidence_inputs: template.evidence_inputs,
        outputs: template.outputs,
        objective: template.objective,
        gpt_role: template.gpt_role,
    }
}

fn evidence_contract(
    evidence: &'static str,
    producer_stage: &'static str,
    consumers: Vec<&'static str>,
    required_for_best_quality: bool,
    status: &'static str,
) -> ScenePlacementEvidenceContract {
    ScenePlacementEvidenceContract {
        evidence,
        producer_stage,
        consumers,
        required_for_best_quality,
        status,
    }
}

fn ablation_axis(
    axis: &'static str,
    selected: &str,
    options: Vec<&'static str>,
    mutual_exclusion_group: &'static str,
    expected_quality_impact: &'static str,
) -> ScenePlacementAblationAxis {
    ScenePlacementAblationAxis {
        axis,
        selected: selected.to_string(),
        options,
        mutual_exclusion_group,
        expected_quality_impact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_bones_geometric_selection() -> ScenePlacementPipelineSelection {
        ScenePlacementPipelineSelection {
            entry_point: ScenePlacementEntryPoint::SceneBuild,
            lift_assets: true,
            composition_mode: SceneCompositionMode::CvGrounded,
            pose_fit: ScenePoseFitMode::RenderedSilhouette,
            canonical_pose: SceneCanonicalPoseMode::Off,
            scale_policy: SceneScalePolicy::AssetPreserving,
            ground_calibration: SceneGroundCalibrationMode::Gpt,
            instance_generation: SceneInstanceGenerationMode::CategoryRepresentative,
            depth_provider: SceneDepthProvider::DepthPro,
            locator: SceneLocatorProvider::LocateAnything,
            segmentation_provider: SceneSegmentationProvider::Sam2,
            feedback: false,
            feedback_iters: 0,
            feedback_rotation_selector: FeedbackRotationSelector::Deterministic,
            feedback_rubric_scorer: FeedbackRubricScorer::Off,
            rotation_fit: SceneRotationFitMode::Off,
            table_pose_refinement: SceneTablePoseRefinementMode::GatedGpt,
            max_pose_candidates: 32,
        }
    }

    #[test]
    fn bare_bones_geometric_strategy_has_dense_depth_pose_optimizer() {
        let plan = scene_placement_pipeline_plan(bare_bones_geometric_selection());
        assert_eq!(plan.quality_profile, "bare_bones_geometric");
        assert!(plan.warnings.is_empty(), "{:#?}", plan.warnings);
        assert_eq!(
            plan.active_pose_optimizer,
            "visible_surface_dense_depth_search_plus_soft_point_refinement_plus_table_refinement"
        );
        assert!(plan.stages.iter().any(|stage| {
            stage.stage == "table_pose_refinement"
                && stage.enabled
                && stage.method == "table_only_geometry_with_gpt_gate"
        }));
        assert!(plan.stages.iter().any(|stage| {
            stage.stage == "metric_depth"
                && stage.enabled
                && stage.method == "depth_pro_f32le_sidecar"
        }));
        assert!(plan.stages.iter().any(|stage| {
            stage.stage == "continuous_refinement"
                && stage.enabled
                && stage.method == "burn_soft_point_surface"
        }));
        assert!(plan.stages.iter().any(|stage| {
            stage.stage == "object_image_synthesis"
                && stage.enabled
                && stage.gpt_role == "image_synthesis"
        }));
        assert!(
            plan.stages
                .iter()
                .any(|stage| { stage.stage == "render_capture_feedback" && !stage.enabled })
        );
    }

    #[test]
    fn heuristic_or_bbox_only_strategy_is_explicitly_degraded() {
        let mut selection = bare_bones_geometric_selection();
        selection.composition_mode = SceneCompositionMode::Heuristic;
        selection.locator = SceneLocatorProvider::Manifest;
        selection.segmentation_provider = SceneSegmentationProvider::BboxPrompt;
        selection.depth_provider = SceneDepthProvider::None;
        selection.pose_fit = ScenePoseFitMode::ProjectedAabb;
        let plan = scene_placement_pipeline_plan(selection);
        assert_eq!(plan.quality_profile, "fallback");
        assert!(!plan.warnings.is_empty());
        assert!(
            plan.stages
                .iter()
                .any(|stage| { stage.stage == "pose_optimizer" && stage.method == "disabled" })
        );
    }
}

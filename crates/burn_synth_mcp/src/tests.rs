use super::*;
use crate::cli::read_json_path;
use crate::prelude::*;
use crate::server::{
    SceneAssetLiftPolicy, scene_asset_lift_policy, scene_object_image_generation_policy,
    validate_scene_asset_outputs_for_policy,
};
use burn_synth::{RuntimeConfig, TrellisComputeProfile};
use burn_synth_grounding::LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT;

fn empty_physical_layout() -> FeedbackPhysicalLayout {
    FeedbackPhysicalLayout {
        pairs: Vec::new(),
        corrections: HashMap::new(),
        object_failures: HashMap::new(),
        hard_failure_count: 0,
        warning_count: 0,
        object_failure_count: 0,
        max_overlap_fraction_smaller: 0.0,
        min_signed_clearance_m: 0.0,
    }
}

fn test_feedback_placement(
    object_id: &str,
    label: &str,
    translation: [f32; 3],
    source_bbox: [f32; 4],
) -> GroundedScenePlacement {
    GroundedScenePlacement {
        entity_id: object_id.to_string(),
        asset_id: object_id.to_string(),
        object_id: object_id.to_string(),
        instance_id: None,
        label: label.to_string(),
        source_bbox,
        contact_pixel: bbox_bottom_center_for_test(source_bbox),
        ground_point: translation,
        translation,
        rotation_y_degrees: 0.0,
        scale: [1.0, 1.0, 1.0],
        local_aabb: SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        },
        target_footprint_m: [1.0, 1.0],
    }
}

fn bbox_bottom_center_for_test(bbox: [f32; 4]) -> [f32; 2] {
    [(bbox[0] + bbox[2]) * 0.5, bbox[3]]
}
use burn_synth_scene::SceneCamera;
use clap::Parser;

#[test]
fn feedback_bsn_serializes_concrete_cache_assets() {
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair".to_string(),
        label: "chair".to_string(),
        aliases: Vec::new(),
        path: None,
        cache_key: Some("chair-cache-key".to_string()),
        reusable: true,
        source_image_path: None,
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        }),
        canonical_frame: None,
        provenance: None,
    };
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![test_feedback_placement(
            "chair_asset",
            "chair",
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4],
        )],
        camera: SceneCamera {
            translation: [0.0, 1.5, -3.0],
            focus: [0.0, 0.5, 0.0],
            yaw: Some(180.0),
            pitch: Some(20.0),
            radius: Some(3.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let commands = vec![json!({
        "type": "spawn_cached",
        "cache_key": "chair-cache-key",
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0, 1.0],
        "scale": [1.0, 1.0, 1.0]
    })];

    let bsn = feedback_bsn_from_commands(&[asset], &layout, &commands).unwrap();
    assert!(bsn.contains("asset chair_asset = \"cache:chair-cache-key\";"));
    let envelope =
        burn_synth_scene::scene_bsn_to_mcp_command_envelope(&bsn, &[], true, None, None).unwrap();
    assert_eq!(envelope["commands"][1]["type"], json!("spawn_cached"));
    assert_eq!(
        envelope["commands"][1]["cache_key"],
        json!("chair-cache-key")
    );
}

#[test]
fn server_args_default_to_balanced_quality_defaults() {
    let args = ServerArgs::parse_from(["burn_synth_mcp"]);
    let config = ServerConfig::from_args(args);
    assert_eq!(config.quality, QualityPreset::Balanced);
    assert_eq!(config.num_steps, 20);
    assert_eq!(config.num_tokens, 1024);
    assert_eq!(config.guidance_scale, 7.0);
    assert_eq!(
        config.scene_segmentation_provider,
        SceneSegmentationProvider::Sam2
    );
    assert_eq!(
        config.scene_segmentation_cdn_base_url.as_deref(),
        Some(DEFAULT_SCENE_SEGMENTATION_CDN_BASE_URL)
    );
    assert!(config.scene_segmentation_allow_download);
    assert_eq!(config.flash_octree_depth, 8);
    assert_eq!(config.flash_min_resolution, 31);
    assert_eq!(config.flash_mini_grid_num, 4);
    assert_eq!(config.flash_num_chunks, 8192);
}

#[test]
fn server_args_accept_scene_ground_command() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "scene-ground",
        "--source-scene-path",
        "/tmp/source.jpg",
        "--manifest",
        "/tmp/manifest.json",
        "--asset-bindings",
        "/tmp/assets.json",
        "--composition-mode",
        "cv-grounded",
        "--depth-provider",
        "depth-pro",
        "--locator",
        "locate-anything",
        "--locate-anything-backend",
        "burn-native",
        "--segmentation-provider",
        "bbox-prompt",
        "--segmentation-precision",
        "bf16",
        "--segmentation-quantization",
        "q8",
        "--pose-fit",
        "rendered-silhouette",
        "--canonical-pose",
        "auto",
        "--scale-policy",
        "bounded-anisotropic",
        "--max-pose-candidates",
        "24",
        "--feedback-iters",
        "5",
        "--feedback-rotation-selector",
        "openai",
        "--feedback-rubric-scorer",
        "openai",
    ]);
    let Some(ServerCommand::SceneGround(command)) = args.command else {
        panic!("expected scene-ground subcommand");
    };
    assert_eq!(command.composition_mode, SceneCompositionMode::CvGrounded);
    assert_eq!(command.depth_provider, SceneDepthProvider::DepthPro);
    assert_eq!(command.locator, SceneLocatorProvider::LocateAnything);
    assert_eq!(
        command.locate_anything_backend,
        Some(LocateAnythingBackend::BurnNative)
    );
    assert_eq!(
        command.segmentation_provider,
        Some(SceneSegmentationProvider::BboxPrompt)
    );
    assert_eq!(
        command.segmentation_precision,
        Some(SceneSegmentationPrecision::Bf16)
    );
    assert_eq!(
        command.segmentation_quantization,
        Some(SceneSegmentationQuantization::Q8)
    );
    assert_eq!(command.pose_fit, ScenePoseFitMode::RenderedSilhouette);
    assert_eq!(command.canonical_pose, SceneCanonicalPoseMode::Auto);
    assert_eq!(command.scale_policy, SceneScalePolicy::BoundedAnisotropic);
    assert_eq!(command.max_pose_candidates, 24);
    assert_eq!(command.feedback_iters, 5);
    assert_eq!(
        command.feedback_rotation_selector,
        FeedbackRotationSelector::Openai
    );
    assert_eq!(command.feedback_rubric_scorer, FeedbackRubricScorer::Openai);
}

#[test]
fn server_args_scene_build_defaults_to_cv_grounded_locate_anything() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "scene-build",
        "--source-scene-path",
        "/tmp/source.jpg",
    ]);
    let Some(ServerCommand::SceneBuild(command)) = args.command else {
        panic!("expected scene-build subcommand");
    };
    assert_eq!(command.composition_mode, SceneCompositionMode::CvGrounded);
    assert_eq!(command.pose_fit, ScenePoseFitMode::RenderedSilhouette);
    assert_eq!(command.canonical_pose, SceneCanonicalPoseMode::Off);
    assert_eq!(command.scale_policy, SceneScalePolicy::AssetPreserving);
    assert_eq!(command.max_pose_candidates, 32);
    assert_eq!(command.depth_provider, SceneDepthProvider::DepthPro);
    assert_eq!(command.locator, SceneLocatorProvider::LocateAnything);
    assert_eq!(command.locate_anything_backend, None);
    assert_eq!(
        command.feedback_rotation_selector,
        FeedbackRotationSelector::Deterministic
    );
    assert!(!command.feedback);
    assert_eq!(command.feedback_iters, DEFAULT_SCENE_FEEDBACK_ITERS);
    assert_eq!(command.rotation_fit, SceneRotationFitMode::Off);
    assert_eq!(command.rotation_fit_max_gpt_rounds, 0);
    assert!((command.rotation_fit_min_mask_iou - 0.45).abs() < 1.0e-6);
    assert!((command.rotation_fit_max_depth_error_m - 0.35).abs() < 1.0e-6);
    assert!(command.rotation_fit_write_artifacts);
    assert_eq!(command.feedback_rubric_scorer, FeedbackRubricScorer::Off);
}

#[test]
fn server_args_scene_ground_defaults_to_bare_bones_geometric_flow() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "scene-ground",
        "--source-scene-path",
        "/tmp/source.jpg",
        "--manifest",
        "/tmp/manifest.json",
        "--asset-bindings",
        "/tmp/asset_bindings.json",
    ]);
    let Some(ServerCommand::SceneGround(command)) = args.command else {
        panic!("expected scene-ground subcommand");
    };
    assert_eq!(command.composition_mode, SceneCompositionMode::CvGrounded);
    assert_eq!(command.pose_fit, ScenePoseFitMode::RenderedSilhouette);
    assert_eq!(command.canonical_pose, SceneCanonicalPoseMode::Off);
    assert_eq!(command.scale_policy, SceneScalePolicy::AssetPreserving);
    assert_eq!(command.depth_provider, SceneDepthProvider::DepthPro);
    assert_eq!(command.locator, SceneLocatorProvider::LocateAnything);
    assert!(!command.feedback);
    assert_eq!(command.rotation_fit, SceneRotationFitMode::Off);
    assert_eq!(command.rotation_fit_max_gpt_rounds, 0);
    assert_eq!(command.feedback_rubric_scorer, FeedbackRubricScorer::Off);
}

#[test]
fn canonical_pose_calibration_applies_bounded_openai_selection() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "mesh chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "single mesh chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.65, 0.65]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "mesh chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some("/tmp/generated_chair.png".to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: "/tmp/chair_crop.jpg".to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": "/tmp/generated_chair.png",
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::Openai,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    run.reports[0].candidates[0].rendered_image_path =
        Some("/tmp/chair_render_candidate_0.png".to_string());
    refresh_canonical_pose_selection_inputs(&mut run);
    assert!(
        run.image_paths
            .iter()
            .any(|path| path == Path::new("/tmp/chair_render_candidate_0.png"))
    );
    assert_eq!(
        run.selection_task["objects"][0]["candidates"][0]["rendered_image_path"],
        json!("/tmp/chair_render_candidate_0.png")
    );
    let candidate = run.reports[0]
        .candidates
        .iter()
        .find(|candidate| (candidate.yaw_offset_degrees - 180.0).abs() <= 1.0e-5)
        .cloned()
        .expect("180 degree chair candidate");

    let report = apply_canonical_pose_openai_selection(
        &mut run,
        &SceneRotationSelectionResponse {
            objects: vec![burn_synth_scene::SceneRotationSelection {
                index: 0,
                candidate_index: candidate.candidate_index,
                confidence: 0.84,
                rationale: "source crop shows the opposite chair face".to_string(),
            }],
        },
    );

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(
        run.asset_bindings[0].canonical_frame.unwrap().source,
        Some(SceneAssetFrameSource::GptVisualSelection)
    );
    assert!(
        (run.asset_bindings[0]
            .canonical_frame
            .unwrap()
            .yaw_offset_degrees
            - 180.0)
            .abs()
            <= 1.0e-5
    );
    assert_eq!(
        run.selection_task["objects"][0]["source_crop_path"],
        json!("/tmp/chair_crop.jpg")
    );
    assert!(
        run.image_paths
            .iter()
            .any(|path| path == Path::new("/tmp/generated_chair.png"))
    );
}

#[test]
fn canonical_pose_calibration_uses_provenance_source_crop_without_requests() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "mesh chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "single mesh chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.65, 0.65]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "mesh chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some("/tmp/generated_chair.png".to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: Some(burn_synth_scene::SceneAssetProvenance {
            run_id: "cached_run".to_string(),
            source_scene_path: "/tmp/source.jpg".to_string(),
            source_object_id: "chair_1".to_string(),
            source_crop_path: Some("/tmp/persisted_chair_crop.jpg".to_string()),
            generated_by: "scene_build_from_image".to_string(),
        }),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": "/tmp/generated_chair.png",
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[],
        &evidence,
    );

    assert_eq!(
        run.reports[0].source_crop_path.as_deref(),
        Some("/tmp/persisted_chair_crop.jpg")
    );
    assert_eq!(
        run.selection_task["objects"][0]["source_crop_path"],
        json!("/tmp/persisted_chair_crop.jpg")
    );
    assert!(
        run.image_paths
            .iter()
            .any(|path| path == Path::new("/tmp/persisted_chair_crop.jpg"))
    );
}

#[test]
fn canonical_pose_calibration_applies_rendered_thumbnail_selection() {
    let root = unique_test_dir("canonical_pose_rendered_selection");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source_path = root.join("source.png");
    let generated_path = root.join("generated.png");
    let wide_render_path = root.join("candidate_wide.png");
    let tall_render_path = root.join("candidate_tall.png");
    write_pose_test_image(&source_path, [22, 5, 42, 59]);
    write_pose_test_image(&generated_path, [21, 5, 43, 59]);
    write_pose_test_image(&wide_render_path, [5, 22, 59, 42]);
    write_pose_test_image(&tall_render_path, [22, 5, 42, 59]);

    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some(generated_path.display().to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": generated_path,
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    run.reports[0].candidates[0].rendered_image_path = Some(wide_render_path.display().to_string());
    let tall_candidate_index = run.reports[0]
        .candidates
        .iter()
        .position(|candidate| (candidate.yaw_offset_degrees - 180.0).abs() <= 1.0e-5)
        .expect("180 candidate");
    run.reports[0].candidates[tall_candidate_index].rendered_image_path =
        Some(tall_render_path.display().to_string());

    let report = apply_canonical_pose_rendered_selection(&mut run);
    let verification =
        canonical_pose_verification_report(SceneCanonicalPoseMode::RenderSweep, &run);

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(verification["status"], json!("verified"));
    assert_eq!(verification["visual_verified"], json!(true));
    assert_eq!(verification["rendered_candidate_count"], json!(2));
    assert_eq!(verification["visual_selected_count"], json!(1));
    assert_eq!(
        run.asset_bindings[0].canonical_frame.unwrap().source,
        Some(SceneAssetFrameSource::VisualRenderSweep)
    );
    assert!(
        (run.asset_bindings[0]
            .canonical_frame
            .unwrap()
            .yaw_offset_degrees
            - 180.0)
            .abs()
            <= 1.0e-5
    );
}

#[test]
fn canonical_pose_rendered_selection_uses_color_edge_descriptor_for_front_back() {
    let root = unique_test_dir("canonical_pose_front_back_descriptor");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source_path = root.join("source.png");
    let generated_path = root.join("generated.png");
    let front_render_path = root.join("candidate_front.png");
    let back_render_path = root.join("candidate_back.png");
    write_asymmetric_pose_test_image(&source_path, false);
    write_asymmetric_pose_test_image(&generated_path, false);
    write_asymmetric_pose_test_image(&front_render_path, true);
    write_asymmetric_pose_test_image(&back_render_path, false);

    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some(generated_path.display().to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": generated_path,
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    run.reports[0].candidates[0].rendered_image_path =
        Some(front_render_path.display().to_string());
    let back_candidate_index = run.reports[0]
        .candidates
        .iter()
        .position(|candidate| (candidate.yaw_offset_degrees - 180.0).abs() <= 1.0e-5)
        .expect("180 candidate");
    run.reports[0].candidates[back_candidate_index].rendered_image_path =
        Some(back_render_path.display().to_string());

    let report = apply_canonical_pose_rendered_selection(&mut run);

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(
        run.reports[0].selected.candidate_index,
        run.reports[0].candidates[back_candidate_index].candidate_index
    );
    assert!(
        (run.reports[0].selected.yaw_offset_degrees - 180.0).abs() <= 1.0e-5,
        "front/back descriptor should select the mirrored 180 degree candidate"
    );
    let selected_metrics = &run.reports[0].candidates[back_candidate_index].metrics["render_similarity"]
        ["visual_descriptor"];
    assert_eq!(selected_metrics["descriptor"], json!("rgb_luma_sobel_edge"));
    assert!(
        selected_metrics["generated"]["score"].as_f64().unwrap()
            > run.reports[0].candidates[0].metrics["render_similarity"]["visual_descriptor"]
                ["generated"]["score"]
                .as_f64()
                .unwrap()
    );
}

#[test]
fn canonical_pose_rendered_selection_uses_depth_mesh_normal_evidence() {
    let root = unique_test_dir("canonical_pose_normal_evidence");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source_path = root.join("source.png");
    let generated_path = root.join("generated.png");
    let front_render_path = root.join("candidate_front.png");
    let back_render_path = root.join("candidate_back.png");
    write_pose_test_image(&source_path, [22, 5, 42, 59]);
    write_pose_test_image(&generated_path, [22, 5, 42, 59]);
    write_pose_test_image(&front_render_path, [22, 5, 42, 59]);
    write_pose_test_image(&back_render_path, [22, 5, 42, 59]);

    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some(generated_path.display().to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": generated_path,
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    run.reports[0].candidates[0].rendered_image_path =
        Some(front_render_path.display().to_string());
    run.reports[0].candidates[0].metrics["normal_evidence"] = json!({
        "similarity": {
            "score": 0.20,
            "descriptor": "depth_normal_vs_mesh_normal",
        }
    });
    let back_candidate_index = run.reports[0]
        .candidates
        .iter()
        .position(|candidate| (candidate.yaw_offset_degrees - 180.0).abs() <= 1.0e-5)
        .expect("180 candidate");
    run.reports[0].candidates[back_candidate_index].rendered_image_path =
        Some(back_render_path.display().to_string());
    run.reports[0].candidates[back_candidate_index].metrics["normal_evidence"] = json!({
        "similarity": {
            "score": 0.92,
            "descriptor": "depth_normal_vs_mesh_normal",
        }
    });

    let report = apply_canonical_pose_rendered_selection(&mut run);

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(
        run.reports[0].selected.candidate_index,
        run.reports[0].candidates[back_candidate_index].candidate_index
    );
    let source_normal_score = run.reports[0].candidates[back_candidate_index].metrics
        ["render_similarity"]["source_normal_score"]
        .as_f64()
        .unwrap();
    assert!((source_normal_score - 0.92).abs() <= 1.0e-5);
    assert!(
        run.reports[0].selected.confidence > 0.55,
        "normal evidence should raise selection confidence above visual-only low-confidence range"
    );
}

#[test]
fn canonical_pose_rendered_fallback_retains_prior_frame() {
    let root = unique_test_dir("canonical_pose_rendered_fallback");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    let source_path = root.join("source.png");
    let generated_path = root.join("generated.png");
    let weak_render_path = root.join("weak_candidate_mask.png");
    write_pose_test_image(&source_path, [22, 5, 42, 59]);
    write_pose_test_image(&generated_path, [22, 5, 42, 59]);
    write_pose_test_image(&weak_render_path, [1, 1, 63, 5]);

    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some(generated_path.display().to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: source_path.display().to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": generated_path,
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    let weak_candidate_index = run.reports[0]
        .candidates
        .iter()
        .position(|candidate| (candidate.yaw_offset_degrees - 180.0).abs() <= 1.0e-5)
        .expect("180 candidate");
    run.reports[0].candidates[weak_candidate_index].rendered_image_path =
        Some(weak_render_path.display().to_string());

    let report = apply_canonical_pose_rendered_selection(&mut run);

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(
        run.reports[0].selected.source,
        SceneAssetFrameSource::AmbiguousFallback
    );
    assert_eq!(run.reports[0].selected.candidate_index, 0);
    assert!(run.reports[0].selected.yaw_offset_degrees.abs() <= 1.0e-5);
    assert_eq!(
        report["applied"][0]["best_measured_candidate_index"],
        json!(weak_candidate_index)
    );
}

#[test]
fn canonical_pose_verification_flags_missing_rendered_thumbnails() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.6, 0.7]),
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "chair".to_string(),
        aliases: vec!["chair".to_string()],
        path: None,
        cache_key: Some("test/chair".to_string()),
        reusable: true,
        source_image_path: Some("/tmp/generated_chair.png".to_string()),
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-0.3, 0.0, -0.35],
            max: [0.3, 1.0, 0.35],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.6, 0.7]))),
        provenance: None,
    };
    let request = ObjectImageRequest {
        object: manifest.objects[0].clone(),
        source_scene_path: "/tmp/source_1024.jpg".to_string(),
        source_crop_path: "/tmp/chair_crop.jpg".to_string(),
        object_reference_image_path: "/tmp/reference.jpg".to_string(),
        prompt: "chair".to_string(),
        candidate_count: 1,
        size: "1024x1024".to_string(),
        quality: "medium".to_string(),
    };
    let selected = vec![json!({
        "object_id": "chair_1",
        "candidate_index": 0,
        "image_path": "/tmp/generated_chair.png",
    })];
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::RenderSweep,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &selected,
        &[request],
        &evidence,
    );
    run.selection_report = json!({
        "selector": "rendered-thumbnail-sweep",
        "applied_count": 0,
        "skipped_count": 1,
        "render_report": {
            "enabled": true,
            "attempted": 0,
            "rendered": 0,
        },
    });

    let verification =
        canonical_pose_verification_report(SceneCanonicalPoseMode::RenderSweep, &run);

    assert_eq!(verification["status"], json!("invalid"));
    assert_eq!(verification["visual_verified"], json!(false));
    assert_eq!(verification["requires_attention"], json!(true));
    assert_eq!(verification["missing_rendered_asset_count"], json!(1));
}

fn write_pose_test_image(path: &Path, rect: [u32; 4]) {
    let mut image = image::RgbaImage::from_pixel(64, 64, image::Rgba([255, 255, 255, 255]));
    for y in rect[1]..rect[3] {
        for x in rect[0]..rect[2] {
            image.put_pixel(x, y, image::Rgba([30, 30, 30, 255]));
        }
    }
    image.save(path).unwrap();
}

fn write_asymmetric_pose_test_image(path: &Path, accent_left: bool) {
    let mut image = image::RgbaImage::from_pixel(64, 64, image::Rgba([255, 255, 255, 255]));
    for y in 5..59 {
        for x in 22..42 {
            image.put_pixel(x, y, image::Rgba([42, 42, 42, 255]));
        }
    }
    let accent = if accent_left { 23..29 } else { 35..41 };
    for y in 12..47 {
        for x in accent.clone() {
            image.put_pixel(x, y, image::Rgba([205, 52, 42, 255]));
        }
    }
    for y in 44..54 {
        for x in 24..40 {
            let shade = if accent_left { x - 24 } else { 39 - x };
            image.put_pixel(x, y, image::Rgba([70 + shade as u8 * 4, 70, 92, 255]));
        }
    }
    image.save(path).unwrap();
}

#[test]
fn images_to_assets_captures_runtime_progress_events() {
    let root = unique_test_dir("asset_runtime_progress");
    fs::create_dir_all(&root).expect("create temp dir");
    let input = root.join("input.png");
    write_pose_test_image(&input, [16, 16, 48, 48]);

    let mut server = McpServer::new(ServerConfig::from_args(ServerArgs::parse_from([
        "burn_synth_mcp",
    ])));
    let result = server
        .call_images_to_assets(ImagesToAssetsToolArgs {
            input_image_paths: vec![input],
            output_dir: Some(root.join("assets")),
            output_paths: None,
            output_format: Some(AssetOutputFormat::Glb),
            rmbg_model: Some(ForegroundModel::Rmbg14),
            synthesis_models: Some(vec![SynthesisModel::Triposg]),
            backend: Some(InferenceBackend::Cpu),
            target_faces: Some(0),
            batch_size: Some(1),
            batch_vram_mb: None,
            trellis_pbr: Some(false),
            trellis_pbr_texture_size: None,
            promote_to_catalog: false,
            dry_run: true,
        })
        .expect("dry-run images_to_assets");

    let events = result["runtime_progress_events"]
        .as_array()
        .expect("runtime progress events array");
    assert!(
        events
            .iter()
            .any(|event| event["kind"] == "run_started" && event["run"] == "mesh"),
        "expected mesh run start in {events:#?}"
    );
    assert!(
        events
            .iter()
            .any(|event| event["kind"] == "stage_completed" && event["stage"] == "mesh.dry_run"),
        "expected mesh dry-run stage completion in {events:#?}"
    );

    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn canonical_pose_calibration_rejects_untrusted_openai_candidate() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_1".to_string(),
            label: "mesh chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.2, 0.2, 0.4, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "single chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let asset = SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair_1".to_string(),
        label: "mesh chair".to_string(),
        aliases: Vec::new(),
        path: None,
        cache_key: None,
        reusable: true,
        source_image_path: None,
        pipeline: None,
        local_aabb: None,
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, None)),
        provenance: None,
    };
    let evidence = manifest_grounding_evidence(&manifest);
    let mut run = build_canonical_pose_calibration(
        SceneCanonicalPoseMode::Openai,
        4,
        &manifest,
        std::slice::from_ref(&asset),
        &[],
        &[],
        &evidence,
    );

    let report = apply_canonical_pose_openai_selection(
        &mut run,
        &SceneRotationSelectionResponse {
            objects: vec![
                burn_synth_scene::SceneRotationSelection {
                    index: 0,
                    candidate_index: 1,
                    confidence: 0.20,
                    rationale: "not enough evidence".to_string(),
                },
                burn_synth_scene::SceneRotationSelection {
                    index: 0,
                    candidate_index: 999,
                    confidence: 0.95,
                    rationale: "invalid candidate".to_string(),
                },
            ],
        },
    );

    assert_eq!(report["applied_count"], json!(0));
    assert_eq!(report["ignored_count"], json!(2));
    assert_eq!(
        run.asset_bindings[0].canonical_frame.unwrap().source,
        Some(SceneAssetFrameSource::VisualRenderSweep)
    );
    assert!(
        run.asset_bindings[0]
            .canonical_frame
            .unwrap()
            .yaw_offset_degrees
            .abs()
            <= 1.0e-5
    );
}

#[test]
fn scene_ground_accepts_rendered_silhouette_pose_fit_mode() {
    let args = ServerArgs::parse_from(["burn_synth_mcp"]);
    let config = ServerConfig::from_args(args);
    let mut server = McpServer::new(config);
    let err = server
        .call_scene_ground(SceneGroundToolArgs {
            source_scene_path: PathBuf::from("/tmp/source.jpg"),
            manifest: SceneObjectManifest {
                source_scene_path: "/tmp/source.jpg".to_string(),
                scene_calibration: None,
                objects: Vec::new(),
            },
            asset_bindings: Vec::new(),
            grounding_evidence: None,
            output_dir: None,
            composition_mode: SceneCompositionMode::CvGrounded,
            pose_fit: ScenePoseFitMode::RenderedSilhouette,
            canonical_pose: SceneCanonicalPoseMode::Auto,
            scale_policy: SceneScalePolicy::AssetPreserving,
            max_pose_candidates: 32,
            save_pose_debug: true,
            ground_calibration: SceneGroundCalibrationMode::DepthHeuristic,
            depth_provider: SceneDepthProvider::None,
            locator: SceneLocatorProvider::Manifest,
            locate_anything_backend: None,
            segmentation_provider: None,
            segmentation_precision: None,
            segmentation_quantization: None,
            clear_existing: true,
            apply: false,
            feedback: false,
            feedback_iters: 0,
            feedback_keep_viewer: false,
            feedback_capture_dir: None,
            feedback_threshold_profile: FeedbackThresholdProfile::Standard,
            feedback_rotation_selector: FeedbackRotationSelector::Deterministic,
            rotation_fit: SceneRotationFitMode::DepthMaskRansac,
            rotation_fit_max_gpt_rounds: 2,
            rotation_fit_min_mask_iou: 0.45,
            rotation_fit_max_depth_error_m: 0.35,
            rotation_fit_write_artifacts: true,
            object_pose_refinement: SceneObjectPoseRefinementMode::GatedGpt,
            object_pose_refinement_set: SceneObjectPoseRefinementSet::TablesAndLargeSeating,
            feedback_rubric_scorer: FeedbackRubricScorer::Off,
        })
        .unwrap_err();

    assert!(!err.contains("pose_fit=rendered-silhouette is not implemented yet"));
}

#[test]
fn server_args_default_to_safe_locate_anything_token_limit() {
    let args = ServerArgs::parse_from(["burn_synth_mcp"]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.locate_anything_in_token_limit,
        LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT as usize
    );
    assert_eq!(
        config.locate_anything_cdn_base_url.as_deref(),
        Some(DEFAULT_LOCATE_ANYTHING_CDN_BASE_URL)
    );
    assert!(config.locate_anything_allow_download);
    assert_eq!(
        config.locate_anything_precision,
        SceneLocateAnythingPrecision::Bf16
    );
}

#[test]
fn server_args_accept_global_locate_anything_backend() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--locate-anything-backend",
        "burn-native",
        "--locate-anything-cache-dir",
        "/tmp/locateanything-cache",
        "--locate-anything-cdn-base-url",
        "https://cdn.example.invalid/model",
        "--locate-anything-allow-download",
        "false",
        "--locate-anything-precision",
        "f16",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.locate_anything_backend,
        LocateAnythingBackend::BurnNative
    );
    assert_eq!(
        config.locate_anything_cache_dir.as_deref(),
        Some(Path::new("/tmp/locateanything-cache"))
    );
    assert_eq!(
        config.locate_anything_cdn_base_url.as_deref(),
        Some("https://cdn.example.invalid/model")
    );
    assert!(!config.locate_anything_allow_download);
    assert_eq!(
        config.locate_anything_precision,
        SceneLocateAnythingPrecision::F16
    );
}

#[test]
fn server_args_accept_global_scene_segmentation_provider() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--scene-segmentation-provider",
        "bbox-prompt",
        "--scene-segmentation-precision",
        "bf16",
        "--scene-segmentation-quantization",
        "q4",
        "--scene-segmentation-model-root",
        "/tmp/sam",
        "--scene-segmentation-cache-dir",
        "/tmp/sam-cache",
        "--scene-segmentation-cdn-base-url",
        "https://cdn.example.invalid/models",
        "--scene-segmentation-allow-download",
        "true",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.scene_segmentation_provider,
        SceneSegmentationProvider::BboxPrompt
    );
    assert_eq!(
        config.scene_segmentation_precision,
        SceneSegmentationPrecision::Bf16
    );
    assert_eq!(
        config.scene_segmentation_quantization,
        SceneSegmentationQuantization::Q4
    );
    assert_eq!(
        config.scene_segmentation_model_root.as_deref(),
        Some(Path::new("/tmp/sam"))
    );
    assert_eq!(
        config.scene_segmentation_cache_dir.as_deref(),
        Some(Path::new("/tmp/sam-cache"))
    );
    assert_eq!(
        config.scene_segmentation_cdn_base_url.as_deref(),
        Some("https://cdn.example.invalid/models")
    );
    assert!(config.scene_segmentation_allow_download);
}

#[test]
fn server_args_accept_cubecl_autotune_controls() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--cubecl-autotune-level",
        "minimal",
        "--cubecl-autotune-cache",
        "global",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.cubecl_autotune_level,
        CubeClAutotuneLevelSetting::Minimal
    );
    assert_eq!(
        config.cubecl_autotune_cache,
        CubeClAutotuneCacheSetting::Global
    );
}

#[test]
fn repo_burn_toml_sets_cubecl_autotune_cache_global() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let config =
        <cubecl::config::CubeClRuntimeConfig as cubecl::config::RuntimeConfig>::from_section_file_path(
            repo_root.join("Burn.toml"),
            "cubecl",
        )
        .expect("parse workspace Burn.toml cubecl section");
    assert!(matches!(
        config.autotune.level,
        cubecl::config::autotune::AutotuneLevel::Balanced
    ));
    assert!(matches!(
        config.autotune.cache,
        cubecl::config::cache::CacheConfig::Global
    ));
}

#[test]
fn locate_anything_burn_native_scene_ground_reuses_runtime_when_enabled() {
    if std::env::var("LOCATE_ANYTHING_MCP_BURN_NATIVE_CACHE_SMOKE").is_err() {
        eprintln!(
            "skipping: set LOCATE_ANYTHING_MCP_BURN_NATIVE_CACHE_SMOKE=1 to run WGPU LocateAnything MCP cache smoke"
        );
        return;
    }
    let Some(repo_root) = find_repo_root_for_test() else {
        eprintln!("skipping LocateAnything MCP cache smoke; repo root not found");
        return;
    };
    let Some(image_path) = std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE").map(PathBuf::from)
    else {
        eprintln!(
            "skipping LocateAnything MCP cache smoke; set LOCATE_ANYTHING_PARITY_IMAGE to the reference scene image"
        );
        return;
    };
    let model_root = repo_root.join("assets/models/LocateAnything-3B");
    if !image_path.exists() || !model_root.join("config.json").exists() {
        eprintln!(
            "skipping LocateAnything MCP cache smoke; missing {} or {}",
            image_path.display(),
            model_root.display()
        );
        return;
    }

    let mut server = McpServer::new(ServerConfig {
        locate_anything_backend: LocateAnythingBackend::BurnNative,
        locate_anything_model_root: model_root,
        ..ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]))
    });
    let manifest = SceneObjectManifest {
        source_scene_path: image_path.display().to_string(),
        scene_calibration: None,
        objects: vec![
            burn_synth_scene::SceneObjectSpec {
                id: "conference_table".to_string(),
                label: "conference table".to_string(),
                aliases: vec!["table".to_string()],
                bbox: [0.386, 0.519, 0.659, 1.0],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("conference_table".to_string()),
                instance_count: 1,
                object_prompt: "conference table".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([3.2, 1.2]),
            },
            burn_synth_scene::SceneObjectSpec {
                id: "conference_chair".to_string(),
                label: "conference chair".to_string(),
                aliases: vec!["chair".to_string()],
                bbox: [0.166, 0.63, 0.36, 1.0],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("conference_chair".to_string()),
                instance_count: 1,
                object_prompt: "conference chair".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: Some([0.65, 0.65]),
            },
        ],
    };
    let assets = vec![
        SceneAssetBinding {
            asset_id: "conference_table_asset".to_string(),
            object_id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("test/conference_table".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: None,
            local_aabb: Some(SceneAssetAabb {
                min: [-1.6, 0.0, -0.6],
                max: [1.6, 0.2, 0.6],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([3.2, 1.2]))),
            provenance: None,
        },
        SceneAssetBinding {
            asset_id: "conference_chair_asset".to_string(),
            object_id: "conference_chair".to_string(),
            label: "conference chair".to_string(),
            aliases: Vec::new(),
            path: None,
            cache_key: Some("test/conference_chair".to_string()),
            reusable: true,
            source_image_path: None,
            pipeline: None,
            local_aabb: Some(SceneAssetAabb {
                min: [-0.32, 0.0, -0.32],
                max: [0.32, 1.1, 0.32],
            }),
            canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([0.64, 0.64]))),
            provenance: None,
        },
    ];
    let root = repo_root.join("tmp/runs").join(format!(
        "{}_locateanything_mcp_burn_native_cache_smoke",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_millis()
    ));
    let first_dir = root.join("first");
    let second_dir = root.join("second");
    let make_args = |output_dir: PathBuf| SceneGroundToolArgs {
        source_scene_path: image_path.clone(),
        manifest: manifest.clone(),
        asset_bindings: assets.clone(),
        grounding_evidence: None,
        output_dir: Some(output_dir),
        composition_mode: SceneCompositionMode::CvGrounded,
        pose_fit: ScenePoseFitMode::ProjectedAabb,
        canonical_pose: SceneCanonicalPoseMode::Auto,
        scale_policy: SceneScalePolicy::AssetPreserving,
        max_pose_candidates: 32,
        save_pose_debug: true,
        ground_calibration: SceneGroundCalibrationMode::DepthHeuristic,
        depth_provider: SceneDepthProvider::None,
        locator: SceneLocatorProvider::LocateAnything,
        locate_anything_backend: Some(LocateAnythingBackend::BurnNative),
        segmentation_provider: None,
        segmentation_precision: None,
        segmentation_quantization: None,
        clear_existing: true,
        apply: false,
        feedback: false,
        feedback_iters: 0,
        feedback_keep_viewer: false,
        feedback_capture_dir: None,
        feedback_threshold_profile: FeedbackThresholdProfile::Standard,
        feedback_rotation_selector: FeedbackRotationSelector::Deterministic,
        rotation_fit: SceneRotationFitMode::DepthMaskRansac,
        rotation_fit_max_gpt_rounds: 2,
        rotation_fit_min_mask_iou: 0.45,
        rotation_fit_max_depth_error_m: 0.35,
        rotation_fit_write_artifacts: true,
        object_pose_refinement: SceneObjectPoseRefinementMode::GatedGpt,
        object_pose_refinement_set: SceneObjectPoseRefinementSet::TablesAndLargeSeating,
        feedback_rubric_scorer: FeedbackRubricScorer::Off,
    };

    server
        .call_scene_ground(make_args(first_dir.clone()))
        .expect("first burn-native scene-ground");
    server
        .call_scene_ground(make_args(second_dir.clone()))
        .expect("second burn-native scene-ground");
    let first_metadata: Value =
        read_json_path(&first_dir.join("locate_anything_burn_native/metadata.json")).unwrap();
    let second_metadata: Value =
        read_json_path(&second_dir.join("locate_anything_burn_native/metadata.json")).unwrap();
    assert_eq!(first_metadata["runtime_cache_hit"], json!(false));
    assert_eq!(second_metadata["runtime_cache_hit"], json!(true));
}

#[test]
fn depth_pro_scene_ground_reuses_runtime_when_enabled() {
    if std::env::var("DEPTH_PRO_MCP_CACHE_SMOKE").is_err() {
        eprintln!("skipping: set DEPTH_PRO_MCP_CACHE_SMOKE=1 to run WGPU DepthPro MCP cache smoke");
        return;
    }
    let Some(repo_root) = find_repo_root_for_test() else {
        eprintln!("skipping DepthPro MCP cache smoke; repo root not found");
        return;
    };
    let Some(image_path) = std::env::var_os("DEPTH_PRO_PARITY_IMAGE")
        .or_else(|| std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE"))
        .map(PathBuf::from)
    else {
        eprintln!("skipping DepthPro MCP cache smoke; set DEPTH_PRO_PARITY_IMAGE to a scene image");
        return;
    };
    if !image_path.exists() {
        eprintln!(
            "skipping DepthPro MCP cache smoke; missing {}",
            image_path.display()
        );
        return;
    }

    let mut server = McpServer::new(ServerConfig::from_args(ServerArgs::parse_from([
        "burn_synth_mcp",
    ])));
    let manifest = SceneObjectManifest {
        source_scene_path: image_path.display().to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "conference_table".to_string(),
            label: "conference table".to_string(),
            aliases: vec!["table".to_string()],
            bbox: [0.386, 0.519, 0.659, 0.96],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("conference_table".to_string()),
            instance_count: 1,
            object_prompt: "conference table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([3.2, 1.2]),
        }],
    };
    let assets = vec![SceneAssetBinding {
        asset_id: "conference_table_asset".to_string(),
        object_id: "conference_table".to_string(),
        label: "conference table".to_string(),
        aliases: Vec::new(),
        path: None,
        cache_key: Some("test/conference_table".to_string()),
        reusable: true,
        source_image_path: None,
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-1.6, 0.0, -0.6],
            max: [1.6, 0.2, 0.6],
        }),
        canonical_frame: Some(SceneAssetFrame::heuristic(0.0, Some([3.2, 1.2]))),
        provenance: None,
    }];
    let root = repo_root.join("tmp/runs").join(format!(
        "{}_depthpro_mcp_cache_smoke",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_millis()
    ));
    let make_args = |output_dir: PathBuf| SceneGroundToolArgs {
        source_scene_path: image_path.clone(),
        manifest: manifest.clone(),
        asset_bindings: assets.clone(),
        grounding_evidence: None,
        output_dir: Some(output_dir),
        composition_mode: SceneCompositionMode::CvGrounded,
        pose_fit: ScenePoseFitMode::ProjectedAabb,
        canonical_pose: SceneCanonicalPoseMode::Auto,
        scale_policy: SceneScalePolicy::AssetPreserving,
        max_pose_candidates: 32,
        save_pose_debug: true,
        ground_calibration: SceneGroundCalibrationMode::DepthHeuristic,
        depth_provider: SceneDepthProvider::DepthPro,
        locator: SceneLocatorProvider::Manifest,
        locate_anything_backend: None,
        segmentation_provider: None,
        segmentation_precision: None,
        segmentation_quantization: None,
        clear_existing: true,
        apply: false,
        feedback: false,
        feedback_iters: 0,
        feedback_keep_viewer: false,
        feedback_capture_dir: None,
        feedback_threshold_profile: FeedbackThresholdProfile::Standard,
        feedback_rotation_selector: FeedbackRotationSelector::Deterministic,
        rotation_fit: SceneRotationFitMode::DepthMaskRansac,
        rotation_fit_max_gpt_rounds: 2,
        rotation_fit_min_mask_iou: 0.45,
        rotation_fit_max_depth_error_m: 0.35,
        rotation_fit_write_artifacts: true,
        object_pose_refinement: SceneObjectPoseRefinementMode::GatedGpt,
        object_pose_refinement_set: SceneObjectPoseRefinementSet::TablesAndLargeSeating,
        feedback_rubric_scorer: FeedbackRubricScorer::Off,
    };
    let first_dir = root.join("first");
    let second_dir = root.join("second");

    server
        .call_scene_ground(make_args(first_dir.clone()))
        .expect("first DepthPro scene-ground");
    server
        .call_scene_ground(make_args(second_dir.clone()))
        .expect("second DepthPro scene-ground");
    let first_metadata: Value =
        read_json_path(&first_dir.join("depth_pro/depth_evidence.json")).unwrap();
    let second_metadata: Value =
        read_json_path(&second_dir.join("depth_pro/depth_evidence.json")).unwrap();
    assert_eq!(first_metadata["runtime_cache_hit"], json!(false));
    assert_eq!(second_metadata["runtime_cache_hit"], json!(true));
    assert!(
        second_metadata["load_ms"].as_f64().unwrap_or(f64::MAX)
            < first_metadata["load_ms"].as_f64().unwrap_or(0.0)
    );
}

#[test]
fn tool_schema_exposes_scene_ground() {
    let tools = tool_defs();
    let names = tools
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    assert!(names.contains(&"scene_ground"));
    let scene_ground = tools
        .iter()
        .find(|tool| tool["name"] == "scene_ground")
        .expect("scene_ground schema");
    assert_eq!(
        scene_ground["inputSchema"]["properties"]["locate_anything_backend"]["enum"],
        json!(["burn-native"])
    );
    assert_eq!(
        scene_ground["inputSchema"]["properties"]["segmentation_provider"]["enum"],
        json!(["none", "bbox-prompt", "sam2", "sam3"])
    );
    assert_eq!(
        scene_ground["inputSchema"]["properties"]["segmentation_precision"]["enum"],
        json!(["f32", "f16", "bf16"])
    );
    assert_eq!(
        scene_ground["inputSchema"]["properties"]["segmentation_quantization"]["enum"],
        json!(["none", "q8", "q4"])
    );
}

#[test]
fn server_args_quality_and_explicit_overrides_map_to_runtime_config() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--quality",
        "fast",
        "--num-steps",
        "18",
        "--guidance-scale",
        "6.5",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(config.quality, QualityPreset::Fast);
    assert_eq!(config.num_steps, 18);
    assert_eq!(config.num_tokens, 512);
    assert_eq!(config.guidance_scale, 6.5);
    assert_eq!(config.flash_octree_depth, 7);
    assert_eq!(config.flash_min_resolution, 31);
    assert_eq!(config.flash_mini_grid_num, 2);
    assert_eq!(config.flash_num_chunks, 4096);

    let runtime = config.runtime_config();
    assert_eq!(runtime.num_steps, 18);
    assert_eq!(runtime.num_tokens, 512);
    assert_eq!(runtime.guidance_scale, 6.5);
    assert_eq!(runtime.flash_extract.octree_depth, 7);
    assert_eq!(runtime.flash_extract.min_resolution, 31);
    assert_eq!(runtime.flash_extract.mini_grid_num, 2);
    assert_eq!(runtime.flash_extract.num_chunks, 4096);
}

#[test]
fn wgpu_trellis_runtime_config_uses_fast_compute_profile() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--backend",
        "wgpu",
        "--synthesis-models",
        "trellis",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.runtime_config().trellis_compute_profile,
        TrellisComputeProfile::WgpuFastF16
    );
}

#[test]
fn non_trellis_runtime_config_keeps_default_trellis_compute_profile() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--backend",
        "wgpu",
        "--synthesis-models",
        "triposg",
    ]);
    let config = ServerConfig::from_args(args);
    assert_eq!(
        config.runtime_config().trellis_compute_profile,
        RuntimeConfig::default().trellis_compute_profile
    );
}

#[test]
fn server_args_accept_scene_build_subcommand() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "--backend",
        "wgpu",
        "--trellis-quality",
        "low",
        "scene-build",
        "--source-scene-path",
        "/tmp/scene.jpg",
        "--output-dir",
        "/tmp/scene-run",
        "--candidate-count",
        "2",
        "--candidate-retry-attempts",
        "3",
        "--candidate-batch-size",
        "1",
        "--batch-size",
        "0",
        "--trellis-pbr",
        "false",
        "--apply",
    ]);
    assert_eq!(args.backend, InferenceBackend::Wgpu);
    assert_eq!(args.trellis_quality, TrellisQuality::Low);
    let Some(ServerCommand::SceneBuild(command)) = args.command else {
        panic!("expected scene-build subcommand");
    };
    assert_eq!(command.source_scene_path, PathBuf::from("/tmp/scene.jpg"));
    assert_eq!(command.output_dir, Some(PathBuf::from("/tmp/scene-run")));
    assert_eq!(command.candidate_count, Some(2));
    assert_eq!(command.candidate_retry_attempts, Some(3));
    assert_eq!(command.candidate_batch_size, Some(1));
    assert_eq!(command.batch_size, Some(0));
    assert!(!command.trellis_pbr);
    assert!(command.apply);
    assert!(!command.feedback);
    assert_eq!(command.feedback_iters, DEFAULT_SCENE_FEEDBACK_ITERS);
    assert_eq!(
        command.feedback_threshold_profile,
        FeedbackThresholdProfile::Standard
    );
}

#[test]
fn scene_build_defaults_to_fast_mesh_only_trellis() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "scene-build",
        "--source-scene-path",
        "/tmp/scene.jpg",
    ]);

    assert_eq!(args.trellis_quality, TrellisQuality::Low);
    let Some(ServerCommand::SceneBuild(command)) = args.command else {
        panic!("expected scene-build subcommand");
    };
    assert!(command.trellis_pbr);
    assert_eq!(command.feedback_iters, DEFAULT_SCENE_FEEDBACK_ITERS);
    assert_eq!(
        command.feedback_threshold_profile,
        FeedbackThresholdProfile::Standard
    );
}

#[test]
fn server_args_accept_scene_feedback_replay_rebuild_flag() {
    let args = ServerArgs::parse_from([
        "burn_synth_mcp",
        "scene-feedback-replay",
        "--output-dir",
        "/tmp/scene-run",
        "--feedback-iters",
        "4",
        "--rebuild-commands-from-grounded-layout",
    ]);
    let Some(ServerCommand::SceneFeedbackReplay(command)) = args.command else {
        panic!("expected scene-feedback-replay subcommand");
    };
    assert_eq!(command.output_dir, PathBuf::from("/tmp/scene-run"));
    assert_eq!(command.feedback_iters, 4);
    assert!(command.rebuild_commands_from_grounded_layout);
}

#[test]
fn dotenv_var_parses_plain_export_and_quoted_values() {
    let root = unique_test_dir("dotenv");
    fs::create_dir_all(&root).expect("create temp dir");
    let path = root.join(".env");
    fs::write(
        &path,
        "\n# comment\nOPENAI_API_KEY=plain-key # local\nexport OPENAI_PROJECT_ID='proj_123'\nOPENAI_BASE_URL=\"https://example.test\"\n",
    )
    .expect("write .env");

    assert_eq!(
        dotenv_var(&path, "OPENAI_API_KEY").as_deref(),
        Some("plain-key")
    );
    assert_eq!(
        dotenv_var(&path, "OPENAI_PROJECT_ID").as_deref(),
        Some("proj_123")
    );
    assert_eq!(
        dotenv_var(&path, "OPENAI_BASE_URL").as_deref(),
        Some("https://example.test")
    );
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn tool_list_includes_batch_splat_and_scene_tools() {
    let tools = tool_defs();
    let names = tools
        .iter()
        .filter_map(|tool| tool["name"].as_str())
        .collect::<Vec<_>>();
    for expected in [
        "images_to_assets",
        "image_to_splat",
        "scene_status",
        "scene_prepare_build",
        "scene_plan_objects",
        "scene_generate_object_images",
        "scene_build_from_image",
        "scene_plan_bsn",
        "scene_apply_bsn",
        "scene_spawn_cached",
        "scene_spawn_path",
        "scene_clear",
        "scene_capture",
        "scene_project_status",
        "scene_compose_assets",
        "scene_validate_layout",
    ] {
        assert!(names.contains(&expected), "missing tool {expected}");
    }
}

#[test]
fn select_scene_candidates_rejects_low_reconstruction_score() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "coffee_table".to_string(),
            label: "coffee table".to_string(),
            aliases: Vec::new(),
            bbox: [0.2, 0.2, 0.8, 0.8],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "white coffee table".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let candidates = vec![burn_synth_scene::ObjectImageCandidate {
        object_id: "coffee_table".to_string(),
        candidate_index: 0,
        image_path: "/tmp/table.png".to_string(),
        raw_image_path: None,
        prompt_hash: "hash".to_string(),
        score: DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE - 0.01,
        provider_request_id: None,
    }];
    let err = select_scene_candidates(&manifest, &candidates).unwrap_err();
    assert!(err.contains("not suitable for TRELLIS/RMBG reconstruction"));
}

#[test]
fn scene_build_tool_schema_exposes_retry_and_artifact_controls() {
    let tools = tool_defs();
    let scene_build = tools
        .iter()
        .find(|tool| tool["name"] == "scene_build_from_image")
        .expect("scene_build_from_image tool");
    let properties = &scene_build["inputSchema"]["properties"];
    for key in [
        "candidate_retry_attempts",
        "candidate_batch_size",
        "min_reconstruction_score",
        "synthesis_models",
        "batch_size",
        "batch_vram_mb",
        "write_artifacts",
        "pose_fit",
        "canonical_pose",
        "max_pose_candidates",
        "save_pose_debug",
        "feedback",
        "feedback_iters",
        "feedback_keep_viewer",
        "feedback_capture_dir",
        "feedback_threshold_profile",
        "feedback_rotation_selector",
        "feedback_rubric_scorer",
    ] {
        assert!(
            properties.get(key).is_some(),
            "scene_build_from_image schema missing {key}"
        );
    }
}

#[test]
fn write_scene_build_artifacts_persists_structured_e2e_outputs() {
    let dir = std::env::temp_dir().join(format!(
        "burn_synth_mcp_artifact_test_{}",
        next_scene_sequence()
    ));
    let _ = fs::remove_dir_all(&dir);
    let response = json!({
        "tool": "scene_build_from_image",
        "manifest_initial": {
            "source_scene_path": "/tmp/scene.jpg",
            "objects": []
        },
        "manifest": {
            "source_scene_path": "/tmp/scene.jpg",
            "objects": []
        },
        "manifest_grounded_for_crops": {
            "source_scene_path": "/tmp/scene.jpg",
            "objects": []
        },
        "pre_generation_grounding_evidence": {
            "source_image_path": "/tmp/scene.jpg",
            "detections": [],
            "camera": {},
            "floor": {},
            "objects": []
        },
        "pre_generation_locate_anything_report": {
            "artifact_dir": "/tmp/scene/locate_anything_burn_native",
            "detections_path": "/tmp/scene/locate_anything_burn_native/detections.json",
            "overlay_path": "/tmp/scene/locate_anything_burn_native/detections_overlay.png",
            "metadata_path": "/tmp/scene/locate_anything_burn_native/metadata.json",
            "elapsed_ms": 1.0,
            "runtime_cache_hit": false,
            "detection_count": 0
        },
        "pre_generation_segmentation_report": {
            "mask_count": 0,
            "runtime_cache_hit": true
        },
        "pre_generation_depth_report": {
            "runtime_cache_hit": true,
            "load_ms": 0.0,
            "infer_ms": 0.0
        },
        "candidates": [],
        "selected_candidates": [],
        "asset_outputs": {
            "items": [
                {
                    "input_image_path": "/tmp/chair.png",
                    "output_path": "/tmp/chair.glb",
                    "cache_key": "chair-cache",
                    "vertices": 12,
                    "faces": 8,
                    "local_aabb": {
                        "min": [-0.5, 0.0, -0.5],
                        "max": [0.5, 1.0, 0.5]
                    }
                }
            ]
        },
        "asset_bindings": [
            {
                "asset_id": "chair_asset",
                "object_id": "chair",
                "label": "conference chair",
                "cache_key": "chair-cache",
                "reusable": true,
                "local_aabb": {
                    "min": [-0.5, 0.0, -0.5],
                    "max": [0.5, 1.0, 0.5]
                },
                "canonical_frame": {
                    "yaw_offset_degrees": 0.0,
                    "footprint_m": [0.85, 0.85],
                    "symmetry": "bilateral",
                    "confidence": 0.58,
                    "source": "descriptor_heuristic"
                }
            }
        ],
        "grounded_layout": {
            "placements": [
                {
                    "object_id": "chair",
                    "asset_id": "chair_asset",
                    "translation": [0.0, 0.0, 1.0],
                    "scale": [1.0, 1.0, 1.0],
                    "target_footprint_m": [0.85, 0.85]
                }
            ],
            "projection_fit": {
                "applied": true,
                "iteration_count": 4,
                "initial_loss": 2.0,
                "final_loss": 1.0,
                "initial_score": 0.33,
                "final_score": 0.50,
                "fit_mode": "projected_aabb_canonical_pose",
                "candidate_count": 1,
                "camera": {
                    "translation": [0.0, 2.0, 5.0],
                    "focus": [0.0, 0.0, 0.0],
                    "vertical_fov_degrees": 70.0,
                    "aspect": 1.777
                },
                "initial_objects": [],
                "objects": [],
                "candidates": [
                    {
                        "index": 0,
                        "object_id": "chair",
                        "label": "conference chair",
                        "stage": "yaw_sweep",
                        "yaw_degrees": 0.0,
                        "total_loss": 1.0,
                        "score": 0.5,
                        "accepted": true
                    }
                ]
            }
        },
        "commands": [{ "type": "clear_scene" }],
        "grounding_contract": {
            "schema_version": 1,
            "source_scene_path": "/tmp/scene.jpg",
            "composition_mode": "cv-grounded",
            "entries": []
        },
        "decision_log": {
            "schema_version": 1,
            "source_scene_path": "/tmp/scene.jpg",
            "entries": []
        },
        "scene_placement_pipeline": {
            "schema_version": 1,
            "quality_profile": "bare_bones_geometric",
            "active_pose_optimizer": "visible_surface_dense_depth_search_plus_soft_point_refinement",
            "stages": [],
            "evidence_contracts": [],
            "ablation_axes": []
        },
        "stage_report": [{ "stage": "generate_object_candidates", "elapsed_ms": 7 }],
        "e2e_summary": {
            "ok": true,
            "elapsed_ms": 12
        },
        "bsn": "synth_scene_v1 {}"
    });

    write_scene_build_artifacts(&dir, &response).unwrap();

    assert!(dir.join("manifest.json").exists());
    assert!(dir.join("manifest_initial.json").exists());
    assert!(dir.join("manifest_grounded_for_crops.json").exists());
    assert!(dir.join("pre_generation_grounding_evidence.json").exists());
    assert!(
        dir.join("pre_generation_locate_anything_report.json")
            .exists()
    );
    assert!(dir.join("pre_generation_segmentation_report.json").exists());
    assert!(dir.join("pre_generation_depth_report.json").exists());
    assert!(dir.join("asset_outputs.json").exists());
    assert!(dir.join("stage_report.json").exists());
    assert!(dir.join("summary.json").exists());
    assert!(dir.join("grounding_contract.json").exists());
    assert!(dir.join("decision_log.json").exists());
    assert!(dir.join("scene_placement_pipeline.json").exists());
    assert!(dir.join("scene.bsn").exists());
    assert!(dir.join("projection_fit_report.json").exists());
    assert!(dir.join("projection_fit_initial.json").exists());
    assert!(dir.join("projection_fit_final.json").exists());
    assert!(dir.join("pose_fit_report.json").exists());
    assert!(dir.join("pose_fit_candidates.json").exists());
    assert!(dir.join("canonical_pose_evidence.json").exists());
    assert!(dir.join("camera_grounding_report.json").exists());
    assert!(dir.join("scene_build_response_structured.json").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn grounding_contract_records_evidence_statuses_and_gpt_roles() {
    let source_scene_path = PathBuf::from("/tmp/scene.jpg");
    let mut response = json!({
        "manifest": {
            "source_scene_path": source_scene_path.display().to_string(),
            "objects": [{"id": "chair"}]
        },
        "object_image_requests": [{"object_id": "chair"}],
        "candidate_generation": {"rejected_objects": []},
        "selected_candidates": [{"object_id": "chair"}],
        "asset_outputs": {
            "items": [{"output_path": "/tmp/chair.glb"}]
        },
        "asset_bindings": [
            {
                "asset_id": "chair_asset",
                "canonical_frame": {
                    "yaw_offset_degrees": 0.0,
                    "confidence": 0.7
                }
            }
        ],
        "grounded_layout": {
            "projection_fit": {
                "applied": true,
                "initial_score": 0.2,
                "final_score": 0.7,
                "final_loss": 0.3,
                "fit_mode": "projected_aabb_canonical_pose"
            }
        },
        "feedback": {
            "enabled": true,
            "accepted": true,
            "accepted_iteration": 1
        },
        "e2e_summary": {
            "ok": true
        }
    });
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": source_scene_path.display().to_string(),
        "feedback_rotation_selector": "openai"
    }))
    .expect("scene build args");
    let detection = burn_synth_scene::Detection {
        label: "chair".to_string(),
        bbox: [0.2, 0.3, 0.4, 0.8],
        point: Some([0.3, 0.8]),
        confidence: Some(0.9),
        source_query: "chair".to_string(),
    };
    let evidence = SceneGroundingEvidence {
        source_image_path: source_scene_path.display().to_string(),
        depth: Some(burn_synth_scene::DepthEvidenceRef {
            provider: "depth-pro".to_string(),
            model: Some("depth-pro".to_string()),
            precision: Some("f16".to_string()),
            artifact_path: None,
            focal_length_px: Some(900.0),
            vertical_fov_degrees: Some(55.0),
            image_size: Some([1600, 900]),
            depth_map_size: Some([1600, 900]),
            floor_sample_count: Some(128),
        }),
        segmentation: None,
        detections: vec![detection.clone()],
        camera: burn_synth_scene::EstimatedCamera {
            focal_length_px: Some(900.0),
            principal_point: Some([800.0, 450.0]),
            image_size: Some([1600, 900]),
            vertical_fov_degrees: Some(55.0),
            confidence: Some(0.9),
        },
        floor: burn_synth_scene::EstimatedFloorPlane {
            normal: [0.0, 1.0, 0.0],
            distance_m: 0.0,
            residual_m: Some(0.04),
            confidence: Some(0.96),
        },
        objects: vec![burn_synth_scene::ObjectGroundingEvidence {
            object_id: "chair".to_string(),
            instance_id: None,
            reuse_group: Some("chair".to_string()),
            detection: Some(detection),
            mask: None,
            asset_id: Some("chair_asset".to_string()),
            contact_pixel: Some([0.3, 0.8]),
            depth_stats: None,
            candidate_floor_contact_rays: Vec::new(),
            metric_contact_point_m: Some([0.0, 0.0, 1.0]),
            target_footprint_m: Some([0.6, 0.6]),
            provenance: vec!["depth_pro".to_string()],
        }],
    };

    attach_scene_grounding_contracts(
        &mut response,
        &args,
        "locate_anything_burn_native",
        &evidence,
        SceneSegmentationProvider::None,
    )
    .expect("attach grounding contracts");

    let entries = response["grounding_contract"]["entries"]
        .as_array()
        .expect("contract entries");
    let detections = entries
        .iter()
        .find(|entry| entry["name"] == "detections")
        .expect("detections contract");
    assert_eq!(detections["status"], json!("verified"));
    let floor = entries
        .iter()
        .find(|entry| entry["name"] == "floor_plane")
        .expect("floor contract");
    assert_eq!(floor["status"], json!("verified"));
    let feedback = entries
        .iter()
        .find(|entry| entry["name"] == "render_feedback")
        .expect("feedback contract");
    assert_eq!(feedback["gpt_role"], json!("bounded_candidate_selection"));
    let decisions = response["decision_log"]["entries"]
        .as_array()
        .expect("decision entries");
    assert!(decisions.iter().any(|entry| {
        entry["decision"] == "scene_pose_fit"
            && entry["source_of_truth"] == "deterministic_visible_surface_mask_depth_fit"
    }));
}

#[test]
fn scene_build_scene_promotion_writes_scene_catalog_snapshot() {
    let root = unique_test_dir("scene_catalog_promotion");
    let source = root.join("source_scene.png");
    let output_dir = root.join("run_20260627T000000Z_scene_test");
    fs::create_dir_all(&output_dir).expect("create output dir");
    fs::write(&source, [137, 80, 78, 71, 13, 10, 26, 10]).expect("write source bytes");
    let mut cache = MeshCache::load_from_root(root.join("cache")).expect("load cache");
    let asset_bindings = vec![SceneAssetBinding {
        asset_id: "chair_asset".to_string(),
        object_id: "chair".to_string(),
        label: "conference chair".to_string(),
        aliases: Vec::new(),
        path: Some("/tmp/chair.glb".to_string()),
        cache_key: Some("chair-cache-key".to_string()),
        reusable: true,
        source_image_path: Some("/tmp/chair.png".to_string()),
        pipeline: Some("trellis".to_string()),
        local_aabb: Some(SceneAssetAabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, 1.0, 0.5],
        }),
        canonical_frame: None,
        provenance: None,
    }];
    let layout = GroundedSceneLayout {
        bsn: "synth_scene_v1 {}".to_string(),
        placements: vec![GroundedScenePlacement {
            entity_id: "chair_0".to_string(),
            asset_id: "chair_asset".to_string(),
            object_id: "chair".to_string(),
            instance_id: None,
            label: "conference chair".to_string(),
            source_bbox: [0.2, 0.3, 0.4, 0.8],
            contact_pixel: [0.3, 0.8],
            ground_point: [1.0, 0.0, 2.0],
            translation: [1.0, 0.0, 2.0],
            rotation_y_degrees: 90.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            },
            target_footprint_m: [1.0, 1.0],
        }],
        camera: SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.0, 0.0],
            yaw: None,
            pitch: None,
            radius: None,
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let response = json!({
        "manifest": {
            "objects": [{"id": "chair"}]
        },
        "grounding_contract": {"entries": []},
        "decision_log": {"entries": []},
        "stage_report": [],
        "token_usage": {},
        "e2e_summary": {
            "ok": true,
            "elapsed_ms": 123,
            "asset_count": 1,
            "placement_count": 1,
            "feedback": {
                "accepted": true,
                "accepted_iteration": 2
            }
        }
    });

    let metadata = promote_scene_build_scene_to_catalog(
        &mut cache,
        &source,
        &output_dir,
        "synth_scene_v1 {}",
        &asset_bindings,
        &layout,
        &response,
    )
    .expect("promote scene");

    assert_eq!(metadata["pipeline"], json!("explicit"));
    assert_eq!(cache.scene_entries().len(), 1);
    assert_eq!(
        cache.scene_entries()[0].metrics.as_ref().unwrap().ok,
        Some(true)
    );
    let scene_key = cache.scene_entries()[0].scene_key.clone();
    let payload = cache
        .load_scene(&scene_key)
        .expect("load scene payload")
        .expect("scene payload");
    assert_eq!(payload.bsn.as_deref(), Some("synth_scene_v1 {}"));
    assert_eq!(payload.world_items.len(), 1);
    assert_eq!(payload.world_items[0].cache_key, "chair-cache-key");
    assert!(payload.asset_bindings.is_some());
    let _ = fs::remove_dir_all(root);
}

#[test]
fn scene_token_usage_summary_groups_openai_usage_by_stage() {
    let provider_metadata = json!({
        "provider": "openai",
        "requests": [
            {
                "operation": "plan_objects",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 25,
                    "total_tokens": 125,
                    "input_tokens_details": {
                        "image_tokens": 60,
                        "text_tokens": 40
                    }
                }
            },
            {
                "operation": "generate_object_images",
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 10,
                    "total_tokens": 40
                }
            },
            {
                "operation": "generate_object_images",
                "usage": null
            }
        ]
    });

    let summary = scene_token_usage_summary(&provider_metadata);

    assert_eq!(summary["total"]["requests"], json!(3));
    assert_eq!(summary["total"]["reported_requests"], json!(2));
    assert_eq!(summary["total"]["unreported_requests"], json!(1));
    assert_eq!(summary["total"]["input_tokens"], json!(130));
    assert_eq!(summary["total"]["output_tokens"], json!(35));
    assert_eq!(summary["total"]["total_tokens"], json!(165));
    assert_eq!(summary["total"]["image_tokens"], json!(60));
    let stages = summary["by_stage"].as_array().expect("stage array");
    assert_eq!(stages.len(), 2);
    assert_eq!(stages[0]["stage"], json!("generate_object_images"));
    assert_eq!(stages[0]["requests"], json!(2));
    assert_eq!(stages[0]["reported_requests"], json!(1));
    assert_eq!(stages[1]["stage"], json!("plan_objects"));
    assert_eq!(stages[1]["total_tokens"], json!(125));
}

#[test]
fn scene_asset_bindings_prefer_promoted_catalog_cache_keys() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_left".to_string(),
            label: "conference chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.1, 0.2, 0.3, 0.7],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "green conference chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let selected = vec![json!({
        "object_id": "chair_left",
        "reuse_group": "chair_left",
        "label": "conference chair",
        "image_path": "/tmp/chair_candidate.png",
        "source_crop_path": "/tmp/chair_source_crop.jpg",
        "candidate_index": 0,
        "score": 0.91,
        "prompt_hash": "abc",
    })];
    let asset_outputs = json!({
        "items": [
            {
                "output_path": "/tmp/chair_candidate_mesh.glb",
                "cache_key": "central-chair-cache-key",
                "synthesis_backend": "trellis",
                "local_aabb": {
                    "min": [-0.5, 0.0, -0.5],
                    "max": [0.5, 1.0, 0.5]
                }
            }
        ]
    });

    let bindings = scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();
    assert_eq!(bindings.len(), 1);
    assert_eq!(
        bindings[0].cache_key.as_deref(),
        Some("central-chair-cache-key")
    );
    assert!(bindings[0].reusable);
    assert_eq!(
        bindings[0].local_aabb.as_ref().map(|aabb| aabb.max[1]),
        Some(1.0)
    );
    assert_eq!(
        bindings[0]
            .provenance
            .as_ref()
            .and_then(|provenance| provenance.source_crop_path.as_deref()),
        Some("/tmp/chair_source_crop.jpg")
    );

    let bsn = "synth_scene_v1 {\nasset chair_left_asset = \"generated:chair_left_asset\";\nspawn chair uses chair_left_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];\n}";
    let plan = parse_scene_bsn(bsn, &bindings).unwrap();
    let commands = scene_plan_to_mcp_commands(&plan, &bindings, true).unwrap();
    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "spawn_cached");
    assert_eq!(commands[1]["cache_key"], json!("central-chair-cache-key"));
}

#[test]
fn scene_asset_bindings_mark_explicit_instances_reusable() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_group".to_string(),
            label: "chair group".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.1, 0.2, 0.8, 0.8],
            instances: vec![
                burn_synth_scene::SceneObjectInstanceSpec {
                    id: Some("left".to_string()),
                    bbox: [0.1, 0.2, 0.25, 0.7],
                    contact: Some([0.18, 0.7]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: None,
                    slot_index: None,
                    target_footprint_m: None,
                },
                burn_synth_scene::SceneObjectInstanceSpec {
                    id: Some("right".to_string()),
                    bbox: [0.65, 0.2, 0.8, 0.7],
                    contact: Some([0.72, 0.7]),
                    rotation_hint_degrees: None,
                    facing_yaw_degrees: None,
                    side: None,
                    slot_index: None,
                    target_footprint_m: None,
                },
            ],
            representative_instance_id: None,
            reuse_group: None,
            instance_count: 1,
            object_prompt: "one reusable chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: None,
        }],
    };
    let selected = vec![json!({
        "object_id": "chair_group",
        "reuse_group": "chair_group",
        "label": "chair group",
        "image_path": "/tmp/chair_candidate.png",
        "candidate_index": 0,
        "score": 0.91,
        "prompt_hash": "abc",
    })];
    let asset_outputs = json!({
        "items": [
            {
                "output_path": "/tmp/chair_candidate_mesh.glb",
                "synthesis_backend": "trellis"
            }
        ]
    });

    let bindings = scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();

    assert_eq!(bindings.len(), 1);
    assert!(bindings[0].reusable);
}

#[test]
fn scene_asset_bindings_expand_reused_groups_to_each_scene_object() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/scene.jpg".to_string(),
        scene_calibration: None,
        objects: vec![
            burn_synth_scene::SceneObjectSpec {
                id: "whiteboard_left".to_string(),
                label: "left whiteboard".to_string(),
                aliases: vec!["whiteboard".to_string()],
                bbox: [0.05, 0.1, 0.35, 0.6],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("whiteboard".to_string()),
                instance_count: 1,
                object_prompt: "whiteboard on a stand".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
            burn_synth_scene::SceneObjectSpec {
                id: "whiteboard_right".to_string(),
                label: "right whiteboard".to_string(),
                aliases: vec!["whiteboard".to_string()],
                bbox: [0.65, 0.1, 0.95, 0.6],
                instances: Vec::new(),
                representative_instance_id: None,
                reuse_group: Some("whiteboard".to_string()),
                instance_count: 1,
                object_prompt: "whiteboard on a stand".to_string(),
                camera_hint: None,
                rotation_hint_degrees: None,
                target_footprint_m: None,
            },
        ],
    };
    let selected = vec![json!({
        "object_id": "whiteboard_left",
        "reuse_group": "whiteboard",
        "label": "left whiteboard",
        "image_path": "/tmp/whiteboard_candidate.png",
        "candidate_index": 0,
        "score": 0.95,
        "prompt_hash": "abc",
    })];
    let asset_outputs = json!({
        "items": [
            {
                "output_path": "/tmp/whiteboard_candidate_mesh.glb",
                "cache_key": "whiteboard-cache-key",
                "synthesis_backend": "trellis",
                "local_aabb": {
                    "min": [-0.5, 0.0, -0.05],
                    "max": [0.5, 1.2, 0.05]
                }
            }
        ]
    });

    let bindings = scene_asset_bindings_from_outputs(&manifest, &selected, &asset_outputs).unwrap();
    assert_eq!(bindings.len(), 2);
    let left = bindings
        .iter()
        .find(|binding| binding.object_id == "whiteboard_left")
        .unwrap();
    let right = bindings
        .iter()
        .find(|binding| binding.object_id == "whiteboard_right")
        .unwrap();
    assert_eq!(left.path, right.path);
    assert_eq!(right.label, "right whiteboard");
    assert!(right.reusable);

    let layout = grounded_scene_layout_for_manifest(&manifest, &bindings).unwrap();
    assert!(layout.bsn.contains("whiteboard_left"));
    assert!(layout.bsn.contains("whiteboard_right"));
}

#[test]
fn scene_asset_quality_failures_gate_trellis_mesh_outputs() {
    let asset_outputs = json!({
        "items": [
            {
                "asset_kind": "mesh",
                "synthesis_backend": "trellis",
                "output_path": "/tmp/chair.glb",
                "mesh_quality_failures": [
                    "position-welded boundary edge ratio 0.4200 exceeds 0.0500"
                ]
            },
            {
                "asset_kind": "mesh",
                "synthesis_backend": "triposg",
                "output_path": "/tmp/legacy.glb",
                "mesh_quality_failures": ["legacy warning"]
            }
        ]
    });

    let failures = scene_asset_quality_failures(&asset_outputs);

    assert_eq!(failures.len(), 1);
    assert!(failures[0].contains("/tmp/chair.glb"));
    assert!(failures[0].contains("boundary edge ratio"));
}

#[test]
fn scene_asset_quality_failures_include_selected_candidate_identity() {
    let selected = vec![json!({
        "object_id": "chair",
        "candidate_index": 2,
        "image_path": "/tmp/chair_candidate_2.png"
    })];
    let asset_outputs = json!({
        "items": [
            {
                "asset_kind": "mesh",
                "synthesis_backend": "trellis",
                "output_path": "/tmp/chair_candidate_2_mesh.glb",
                "mesh_quality_failures": [
                    "position-welded boundary edge ratio 0.1226 exceeds 0.0500"
                ]
            }
        ]
    });

    let failures = scene_asset_quality_failures_with_selected(&asset_outputs, &selected);

    assert_eq!(failures.len(), 1);
    assert_eq!(failures[0].object_id, "chair");
    assert_eq!(failures[0].candidate_index, 2);
    assert!(failures[0].message().contains("chair_candidate_2_mesh.glb"));
}

#[test]
fn scene_asset_quality_gate_waives_narrow_chair_leg_boundary_false_positive() {
    let selected = vec![json!({
        "object_id": "chair_group",
        "reuse_group": "conference_chair",
        "label": "reusable conference chair",
        "candidate_index": 0,
    })];
    let asset_outputs = json!({
        "items": [
            {
                "asset_kind": "mesh",
                "synthesis_backend": "trellis",
                "output_path": "/tmp/chair_candidate_0_mesh.glb",
                "mesh_quality": {
                    "position_welded_connectivity": {
                        "boundary_edge_ratio": 0.0766,
                        "non_manifold_edges": 0,
                        "tiny_components_le_16_faces": 0
                    }
                },
                "mesh_quality_failures": [
                    "position-welded boundary edge ratio 0.0766 exceeds 0.0500"
                ]
            }
        ]
    });

    let failures = scene_asset_quality_failures_with_selected(&asset_outputs, &selected);

    assert!(failures.is_empty());
}

#[test]
fn scene_asset_quality_gate_rejects_severely_fragmented_chair() {
    let selected = vec![json!({
        "object_id": "chair_group",
        "reuse_group": "conference_chair",
        "label": "reusable conference chair",
        "candidate_index": 1,
    })];
    let asset_outputs = json!({
        "items": [
            {
                "asset_kind": "mesh",
                "synthesis_backend": "trellis",
                "output_path": "/tmp/chair_candidate_1_mesh.glb",
                "mesh_quality": {
                    "position_welded_connectivity": {
                        "boundary_edge_ratio": 0.2704,
                        "non_manifold_edges": 1,
                        "tiny_components_le_16_faces": 1
                    }
                },
                "mesh_quality_failures": [
                    "position-welded boundary edge ratio 0.2704 exceeds 0.0500"
                ]
            }
        ]
    });

    let failures = scene_asset_quality_failures_with_selected(&asset_outputs, &selected);

    assert_eq!(failures.len(), 1);
    assert!(failures[0].message().contains("0.2704"));
}

#[test]
fn scene_build_summary_marks_mesh_quality_failures_not_ok() {
    let response = json!({
        "manifest": {
            "source_scene_path": "/tmp/scene.jpg"
        },
        "candidate_generation": {
            "rejected_objects": []
        },
        "mesh_quality_failures": [
            "/tmp/chair.glb: position-welded boundary edge ratio 0.1226 exceeds 0.0500"
        ],
        "failed_stage": "images_to_assets.mesh_quality_gate",
        "asset_lift_attempts": [
            {
                "attempt_index": 0,
                "mesh_quality_failures": [
                    "/tmp/chair.glb: position-welded boundary edge ratio 0.1226 exceeds 0.0500"
                ]
            }
        ]
    });

    let summary = scene_build_summary(&response, Duration::from_millis(42));

    assert_eq!(summary["ok"], json!(false));
    assert_eq!(
        summary["failed_stage"],
        json!("images_to_assets.mesh_quality_gate")
    );
    assert_eq!(summary["asset_lift_attempts"][0]["attempt_index"], json!(0));
}

#[test]
fn scene_build_summary_marks_failed_feedback_not_ok() {
    let response = json!({
        "manifest": {
            "source_scene_path": "/tmp/scene.jpg"
        },
        "candidate_generation": {
            "rejected_objects": []
        },
        "mesh_quality_failures": [],
        "feedback": {
            "enabled": true,
            "accepted": false,
            "accepted_iteration": null,
            "capture_dir": "/tmp/scene/iterations"
        },
        "next_action": {
            "kind": "composition_feedback_failed",
            "report": "/tmp/scene/iterations/feedback_report.md"
        }
    });

    let summary = scene_build_summary(&response, Duration::from_millis(42));

    assert_eq!(summary["ok"], json!(false));
    assert_eq!(summary["feedback"]["gate_passed"], json!(false));
    assert_eq!(
        summary["next_action"]["kind"],
        json!("composition_feedback_failed")
    );
}

#[test]
fn cached_asset_outputs_preserve_selected_candidate_order() {
    let mut cache = HashMap::new();
    let lifted = vec![
        json!({"object_id": "chair", "candidate_index": 1}),
        json!({"object_id": "table", "candidate_index": 0}),
    ];
    let outputs = json!({
        "items": [
            {"output_path": "/tmp/chair_1.glb"},
            {"output_path": "/tmp/table_0.glb"}
        ]
    });
    cache_scene_asset_outputs(&mut cache, &lifted, &outputs).unwrap();

    let selected = vec![
        json!({"object_id": "table", "candidate_index": 0}),
        json!({"object_id": "chair", "candidate_index": 1}),
    ];
    let merged = scene_cached_asset_outputs_for_selected(&selected, &cache).unwrap();
    let items = merged["items"].as_array().unwrap();

    assert_eq!(items[0]["output_path"], json!("/tmp/table_0.glb"));
    assert_eq!(items[1]["output_path"], json!("/tmp/chair_1.glb"));
}

#[test]
fn cache_asset_outputs_rejects_count_mismatch() {
    let mut cache = HashMap::new();
    let lifted = vec![json!({"object_id": "chair", "candidate_index": 1})];
    let outputs = json!({"items": []});

    let err = cache_scene_asset_outputs(&mut cache, &lifted, &outputs).unwrap_err();

    assert!(err.contains("asset output count"));
}

#[test]
fn scene_commands_with_cache_reload_preserves_clear_first() {
    let commands = scene_commands_with_cache_reload(vec![
        json!({ "type": "clear_scene" }),
        json!({ "type": "spawn_cached", "cache_key": "chair" }),
        json!({ "type": "spawn_cached", "cache_key": "table" }),
    ]);

    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "reload_cache");
    assert_eq!(commands[2]["type"], "spawn_cached");
    assert_eq!(commands.len(), 4);
}

#[test]
fn scene_commands_with_asset_local_aabbs_enriches_saved_replay_commands() {
    let commands = vec![
        json!({ "type": "clear_scene" }),
        json!({
            "type": "spawn_path",
            "cache_key": "table_asset",
            "path": "/tmp/table.glb",
        }),
    ];
    let assets = vec![SceneAssetBinding {
        asset_id: "table_asset".to_string(),
        object_id: "table".to_string(),
        label: "table".to_string(),
        aliases: Vec::new(),
        path: Some("/tmp/table.glb".to_string()),
        cache_key: None,
        reusable: false,
        source_image_path: None,
        pipeline: None,
        local_aabb: Some(SceneAssetAabb {
            min: [-1.0, -0.1, -0.5],
            max: [1.0, 0.1, 0.5],
        }),
        canonical_frame: None,
        provenance: None,
    }];

    let enriched = scene_commands_with_asset_local_aabbs(commands, &assets);

    let min = enriched[1]["local_aabb"]["min"]
        .as_array()
        .expect("min array");
    let max = enriched[1]["local_aabb"]["max"]
        .as_array()
        .expect("max array");
    assert!((min[0].as_f64().unwrap() + 1.0).abs() < 1.0e-6);
    assert!((min[1].as_f64().unwrap() + 0.1).abs() < 1.0e-6);
    assert!((min[2].as_f64().unwrap() + 0.5).abs() < 1.0e-6);
    assert!((max[0].as_f64().unwrap() - 1.0).abs() < 1.0e-6);
    assert!((max[1].as_f64().unwrap() - 0.1).abs() < 1.0e-6);
    assert!((max[2].as_f64().unwrap() - 0.5).abs() < 1.0e-6);
}

#[test]
fn scene_interaction_lock_command_uses_viewer_control_protocol() {
    let command = scene_interaction_lock_command(true, "iterative scene composition");

    assert_eq!(command["type"], json!("set_interaction_lock"));
    assert_eq!(command["locked"], json!(true));
    assert_eq!(command["reason"], json!("iterative scene composition"));
}

#[test]
fn feedback_deltas_adjust_spawn_and_camera_commands() {
    let commands = vec![
        json!({ "type": "clear_scene" }),
        json!({
            "type": "spawn_cached",
            "cache_key": "chair",
            "translation": [1.0, 0.5, 2.0],
            "scale": [1.0, 1.0, 1.0],
            "rotation": [0.0, 0.0, 0.0, 1.0]
        }),
        json!({
            "type": "set_camera",
            "translation": [0.0, 2.0, 5.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
            "focus": [0.0, 0.0, 0.0],
            "yaw": 180.0,
            "pitch": 25.0,
            "radius": 5.0,
            "vertical_fov": 72.0
        }),
    ];
    let deltas = json!({
        "objects": [{
            "index": 0,
            "translation_delta": [0.25, 0.0, -0.5],
            "scale_multiplier": 1.2,
            "yaw_delta_degrees": 12.0
        }],
        "camera": {
            "radius_multiplier": 0.9
        }
    });

    let adjusted = apply_feedback_deltas_to_commands_with_policy(
        &commands,
        &deltas,
        SceneScalePolicy::FreeAnisotropic,
    )
    .unwrap();

    assert_eq!(adjusted[1]["translation"], json!([1.25, 0.5, 1.5]));
    let adjusted_scale = adjusted[1]["scale"]
        .as_array()
        .expect("spawn command keeps scale array");
    for component in adjusted_scale {
        let component = component.as_f64().expect("scale component is numeric");
        assert!((component - 1.2).abs() <= 1.0e-5);
    }
    let adjusted_yaw = quat_y_degrees(json_array4(&adjusted[1]["rotation"]).unwrap());
    assert!((adjusted_yaw - 12.0).abs() <= 1.0e-4);
    assert_eq!(adjusted[2]["radius"], json!(4.5));
}

#[test]
fn feedback_deltas_clamp_spawn_translation_to_ground_anchor() {
    let commands = vec![json!({
        "type": "spawn_cached",
        "cache_key": "chair",
        "translation": [0.4, 0.0, 0.0],
        "scale": [1.0, 1.0, 1.0],
        "rotation": [0.0, 0.0, 0.0, 1.0]
    })];
    let deltas = json!({
        "objects": [{
            "index": 0,
            "translation_delta": [1.0, 0.0, 0.0],
            "scale_multiplier": 1.0,
            "yaw_delta_degrees": 0.0,
            "ground_anchor_point": [0.0, 0.0, 0.0],
            "ground_anchor_max_drift_m": 0.6
        }]
    });

    let adjusted = apply_feedback_deltas_to_commands_with_policy(
        &commands,
        &deltas,
        SceneScalePolicy::FreeAnisotropic,
    )
    .unwrap();
    let translation = json_array3(&adjusted[0]["translation"]).unwrap();

    assert!((translation[0] - 0.6).abs() <= 1.0e-5);
    assert_eq!(translation[1], 0.0);
    assert_eq!(translation[2], 0.0);
}

#[test]
fn feedback_layout_deltas_prefer_target_ground_point_as_anchor() {
    let metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "label": "chair",
            "cache_key": "chair-cache",
            "translation_delta": [0.75, 0.0, -0.25],
            "scale_multiplier": 1.0,
            "yaw_delta_degrees": 0.0,
            "ground_anchor_point": [0.0, 0.0, 0.0],
            "target_ground_point": [2.0, 0.0, -1.0],
            "ground_anchor_max_drift_m": 0.6
        }]
    });

    let deltas = feedback_layout_deltas_with_policy(&metrics, SceneScalePolicy::FreeAnisotropic);

    assert_eq!(
        deltas["objects"][0]["ground_anchor_point"],
        json!([2.0, 0.0, -1.0])
    );
}

#[test]
fn projected_collision_correction_clears_anchor_clamp() {
    let mut deltas = vec![FeedbackDeltaDraft {
        index: json!(0),
        translation_delta: [0.0, 0.0, 0.0],
        scale_multiplier: 1.0,
        scale_multiplier_xyz: None,
        scale_group_key: None,
        scale_source: "object_projection",
        yaw_delta_degrees: json!(0.0),
        max_yaw_delta_degrees: 30.0,
        ground_anchor_point: Some([0.0, 0.0, 0.0]),
        ground_anchor_max_drift_m: Some(0.25),
    }];
    let mut footprints = vec![Some(FeedbackFootprint {
        index: 0,
        kind: FeedbackPhysicalKind::Seating,
        rect: FootprintRect {
            min_x: -0.2,
            min_z: -0.2,
            max_x: 0.2,
            max_z: 0.2,
        },
    })];

    assert!(apply_projected_delta(
        &mut deltas,
        &mut footprints,
        0,
        [0.4, 0.0, 0.0],
        1.0
    ));

    assert_eq!(deltas[0].ground_anchor_point, None);
    assert_eq!(deltas[0].ground_anchor_max_drift_m, None);
}

#[test]
fn feedback_deltas_apply_axis_scale_for_table_projection() {
    let commands = vec![json!({
        "type": "spawn_cached",
        "cache_key": "table",
        "translation": [0.0, 0.0, 0.0],
        "scale": [2.0, 0.5, 4.0],
    })];
    let deltas = json!({
        "objects": [{
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 1.0,
            "scale_multiplier_xyz": [1.2, 1.0, 0.9]
        }]
    });

    let adjusted = apply_feedback_deltas_to_commands_with_policy(
        &commands,
        &deltas,
        SceneScalePolicy::FreeAnisotropic,
    )
    .unwrap();

    let scale = json_array3(&adjusted[0]["scale"]).unwrap();
    assert!((scale[0] - 2.4).abs() <= 1.0e-5);
    assert!((scale[1] - 0.5).abs() <= 1.0e-5);
    assert!((scale[2] - 3.6).abs() <= 1.0e-5);
}

#[test]
fn feedback_deltas_asset_preserving_policy_removes_table_axis_outlier() {
    let commands = vec![json!({
        "type": "spawn_cached",
        "cache_key": "table",
        "translation": [0.0, 0.0, 0.0],
        "scale": [5.00385, 0.9624, 1.03558],
    })];
    let deltas = json!({
        "objects": [{
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 1.0,
        }]
    });

    let adjusted = apply_feedback_deltas_to_commands_with_policy(
        &commands,
        &deltas,
        SceneScalePolicy::AssetPreserving,
    )
    .unwrap();

    let scale = json_array3(&adjusted[0]["scale"]).unwrap();
    assert!((scale[0] - 1.03558).abs() <= 1.0e-5);
    assert_eq!(scale[0], scale[1]);
    assert_eq!(scale[1], scale[2]);
}

#[test]
fn feedback_deltas_do_not_emit_axis_scale_for_skinny_table_projection() {
    let metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "conference_table",
            "label": "white rectangular conference table",
            "cache_key": "table-cache",
            "expected_bbox": [0.30, 0.48, 0.65, 0.96],
            "observed_bbox": [0.40, 0.44, 0.58, 0.95],
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 1.22,
            "yaw_delta_degrees": 0.0
        }]
    });

    let deltas = feedback_layout_deltas_with_policy(&metrics, SceneScalePolicy::FreeAnisotropic);
    let object = &deltas["objects"][0];

    assert!(object["scale_multiplier_xyz"].is_null());
    assert_eq!(object["scale_source"], json!("object_projection"));
    assert_eq!(object["scale_multiplier"], json!(1.18));
}

#[test]
fn feedback_deltas_default_to_asset_preserving_table_scale() {
    let metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "conference_table",
            "label": "white rectangular conference table",
            "cache_key": "table-cache",
            "expected_bbox": [0.30, 0.48, 0.65, 0.96],
            "observed_bbox": [0.40, 0.44, 0.58, 0.95],
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 1.22,
            "yaw_delta_degrees": 0.0
        }]
    });

    let deltas = feedback_layout_deltas(&metrics);
    let object = &deltas["objects"][0];

    assert!(object["scale_multiplier_xyz"].is_null());
    assert_eq!(object["scale_multiplier"], json!(1.0));
    assert_eq!(object["scale_source"], json!("object_projection"));
}

#[test]
fn feedback_deltas_shrink_pathological_edge_cropped_table_projection() {
    let metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "conference_table",
            "label": "white rectangular conference table",
            "cache_key": "table-cache",
            "expected_bbox": [0.38, 0.52, 0.66, 1.0],
            "observed_bbox": [-0.02, 0.37, 1.22, 1.70],
            "source_edge_cropped": true,
            "bbox_overscan": 0.92,
            "max_bbox_overscan": 0.02,
            "area_log2_error": 3.6,
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 0.82,
            "yaw_delta_degrees": 0.0
        }]
    });

    let deltas = feedback_layout_deltas(&metrics);
    let object = &deltas["objects"][0];

    assert!(object["scale_multiplier_xyz"].is_null());
    assert_eq!(object["scale_multiplier"], json!(0.82));
    assert_eq!(object["scale_source"], json!("object_projection"));
}

#[test]
fn feedback_deltas_normalize_existing_reused_command_scales() {
    let commands = vec![
        json!({
            "type": "spawn_cached",
            "cache_key": "chair-cache",
            "translation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
        }),
        json!({
            "type": "spawn_cached",
            "cache_key": "chair-cache",
            "translation": [1.0, 0.0, 0.0],
            "scale": [2.0, 1.5, 1.0],
        }),
        json!({
            "type": "spawn_cached",
            "cache_key": "table-cache",
            "translation": [0.0, 0.0, 1.0],
            "scale": [0.75, 0.75, 0.75],
        }),
    ];
    let deltas = json!({
        "objects": [
            { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 },
            { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 },
            { "translation_delta": [0.0, 0.0, 0.0], "scale_multiplier": 1.0 }
        ]
    });

    let adjusted = apply_feedback_deltas_to_commands(&commands, &deltas).unwrap();

    assert_eq!(adjusted[0]["scale"], json!([1.25, 1.25, 1.25]));
    assert_eq!(adjusted[1]["scale"], json!([1.25, 1.25, 1.25]));
    assert_eq!(adjusted[2]["scale"], json!([0.75, 0.75, 0.75]));
}

#[test]
fn feedback_deltas_share_scale_for_reused_instances() {
    let metrics = json!({
        "objects": [
            {
                "index": 0,
                "object_id": "chair",
                "cache_key": "chair-cache",
                "expected_bbox": [0.1, 0.1, 0.2, 0.3],
                "observed_bbox": [0.1, 0.1, 0.3, 0.5],
                "translation_delta": [0.0, 0.0, 0.0],
                "scale_multiplier": 0.82,
                "yaw_delta_degrees": 0.0
            },
            {
                "index": 1,
                "object_id": "chair",
                "cache_key": "chair-cache",
                "expected_bbox": [0.5, 0.1, 0.6, 0.3],
                "observed_bbox": [0.5, 0.1, 0.55, 0.2],
                "translation_delta": [0.0, 0.0, 0.0],
                "scale_multiplier": 1.22,
                "yaw_delta_degrees": 0.0
            },
            {
                "index": 2,
                "object_id": "table",
                "cache_key": "table-cache",
                "expected_bbox": [0.2, 0.4, 0.8, 0.6],
                "observed_bbox": [0.2, 0.4, 0.8, 0.6],
                "translation_delta": [0.0, 0.0, 0.0],
                "scale_multiplier": 0.95,
                "yaw_delta_degrees": 0.0
            }
        ]
    });

    let deltas = feedback_layout_deltas(&metrics);
    let objects = deltas["objects"].as_array().unwrap();
    let chair_scale_a = objects[0]["scale_multiplier"].as_f64().unwrap();
    let chair_scale_b = objects[1]["scale_multiplier"].as_f64().unwrap();
    let table_scale = objects[2]["scale_multiplier"].as_f64().unwrap();

    assert!((chair_scale_a - 1.02).abs() <= 1.0e-6);
    assert!((chair_scale_b - 1.02).abs() <= 1.0e-6);
    assert!((table_scale - 1.0).abs() <= 1.0e-6);
    assert_eq!(objects[0]["scale_group_key"], json!("chair-cache"));
    assert_eq!(objects[1]["scale_source"], json!("repeated_instance_group"));
    assert_eq!(objects[2]["scale_source"], json!("object_projection"));
}

#[test]
fn feedback_deltas_project_simultaneous_chair_moves_out_of_overlap() {
    let left_rect = FootprintRect {
        min_x: -1.0,
        min_z: -0.2,
        max_x: -0.6,
        max_z: 0.2,
    };
    let right_rect = FootprintRect {
        min_x: 0.6,
        min_z: -0.2,
        max_x: 1.0,
        max_z: 0.2,
    };
    let metrics = json!({
        "thresholds": {
            "max_seating_seating_overlap_fraction": 0.10,
            "max_seating_seating_penetration_m": 0.05
        },
        "objects": [
            {
                "index": 0,
                "object_id": "chair_left",
                "cache_key": "chair-cache",
                "expected_bbox": [0.2, 0.6, 0.35, 0.95],
                "observed_bbox": [0.2, 0.6, 0.35, 0.95],
                "translation_delta": [0.8, 0.0, 0.0],
                "scale_multiplier": 1.0,
                "yaw_delta_degrees": 0.0,
                "physical_kind": "seating",
                "world_footprint": {
                    "min_x": left_rect.min_x,
                    "min_z": left_rect.min_z,
                    "max_x": left_rect.max_x,
                    "max_z": left_rect.max_z
                }
            },
            {
                "index": 1,
                "object_id": "chair_right",
                "cache_key": "chair-cache",
                "expected_bbox": [0.65, 0.6, 0.8, 0.95],
                "observed_bbox": [0.65, 0.6, 0.8, 0.95],
                "translation_delta": [-0.8, 0.0, 0.0],
                "scale_multiplier": 1.0,
                "yaw_delta_degrees": 0.0,
                "physical_kind": "seating",
                "world_footprint": {
                    "min_x": right_rect.min_x,
                    "min_z": right_rect.min_z,
                    "max_x": right_rect.max_x,
                    "max_z": right_rect.max_z
                }
            }
        ]
    });

    let deltas = feedback_layout_deltas(&metrics);
    let objects = deltas["objects"].as_array().unwrap();
    let left_delta = json_array3(&objects[0]["translation_delta"]).unwrap();
    let right_delta = json_array3(&objects[1]["translation_delta"]).unwrap();
    let projected_left = left_rect.translated(left_delta);
    let projected_right = right_rect.translated(right_delta);

    assert!(left_delta[0] < 0.8);
    assert!(right_delta[0] > -0.8);
    assert!(
        projected_left.signed_clearance(projected_right)
            >= -FeedbackThresholdProfile::Standard
                .thresholds()
                .max_seating_seating_penetration_m
                - 1.0e-4
    );
}

#[test]
fn feedback_metrics_fail_chair_contained_inside_table_footprint() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let table = test_feedback_placement(
        "conference_table",
        "conference table",
        [0.0, 0.0, 0.0],
        [0.30, 0.40, 0.70, 0.70],
    );
    let chair = test_feedback_placement(
        "conference_chair",
        "conference chair",
        [0.0, 0.0, 0.0],
        [0.45, 0.55, 0.55, 0.85],
    );
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![table, chair],
        camera: SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.0, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(5.0),
            vertical_fov_degrees: Some(72.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [
            {
                "cache_key": "table",
                "screen_bbox": [0.30, 0.40, 0.70, 0.70],
                "screen_contact": [0.50, 0.55],
                "world_aabb": {
                    "min": [-1.5, 0.0, -0.6],
                    "max": [1.5, 0.4, 0.6]
                }
            },
            {
                "cache_key": "chair",
                "screen_bbox": [0.45, 0.55, 0.55, 0.85],
                "screen_contact": [0.50, 0.85],
                "world_aabb": {
                    "min": [-0.25, 0.0, -0.25],
                    "max": [0.25, 1.0, 0.25]
                }
            }
        ],
        "camera": { "radius": 5.0 }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/iter.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert!(!metrics["passed"].as_bool().unwrap());
    assert_eq!(metrics["projection_passed"], json!(true));
    assert_eq!(metrics["physical_passed"], json!(false));
    assert_eq!(metrics["physical_layout"]["hard_failure_count"], json!(1));
    assert_eq!(
        metrics["physical_layout"]["pairs"][0]["failure_reasons"][0],
        json!("seating_center_inside_table")
    );
    assert_eq!(metrics["objects"][1]["physical_passed"], json!(false));
    let deltas = feedback_layout_deltas(&metrics);
    let translation_delta = json_array3(&deltas["objects"][1]["translation_delta"]).unwrap();
    assert!(translation_delta[0].abs() + translation_delta[2].abs() > 0.1);
}

#[test]
fn feedback_metrics_fail_reused_asset_instances_with_different_scales() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let mut left = test_feedback_placement(
        "chair_group",
        "conference chair",
        [-1.0, 0.0, 0.0],
        [0.20, 0.50, 0.35, 0.80],
    );
    left.asset_id = "chair_asset".to_string();
    left.instance_id = Some("left_chair".to_string());
    let mut right = test_feedback_placement(
        "chair_group",
        "conference chair",
        [1.0, 0.0, 0.0],
        [0.65, 0.50, 0.80, 0.80],
    );
    right.asset_id = "chair_asset".to_string();
    right.instance_id = Some("right_chair".to_string());
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![left, right],
        camera: SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.0, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(5.0),
            vertical_fov_degrees: Some(72.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "world_items": [
            {
                "cache_key": "chair-cache",
                "scale": [1.00, 1.00, 1.00],
                "translation": [-1.0, 0.0, 0.0]
            },
            {
                "cache_key": "chair-cache",
                "scale": [1.20, 1.20, 1.20],
                "translation": [1.0, 0.0, 0.0]
            }
        ],
        "projected_items": [
            {
                "cache_key": "chair-cache",
                "screen_bbox": [0.20, 0.50, 0.35, 0.80],
                "screen_contact": [0.275, 0.80],
                "world_aabb": {
                    "min": [-1.35, 0.0, -0.35],
                    "max": [-0.65, 1.0, 0.35]
                }
            },
            {
                "cache_key": "chair-cache",
                "screen_bbox": [0.65, 0.50, 0.80, 0.80],
                "screen_contact": [0.725, 0.80],
                "world_aabb": {
                    "min": [0.65, 0.0, -0.35],
                    "max": [1.35, 1.0, 0.35]
                }
            }
        ],
        "camera": { "radius": 5.0 }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/iter.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["projection_passed"], json!(true));
    assert_eq!(metrics["physical_passed"], json!(true));
    assert_eq!(metrics["scale_consistency_passed"], json!(false));
    assert_eq!(metrics["scale_consistency"]["hard_failure_count"], json!(1));
    assert!(!metrics["passed"].as_bool().unwrap());
    assert!(
        metrics["objects"][0]["physical_failures"][0]
            .as_str()
            .unwrap()
            .contains("reused asset scale mismatch")
    );
}

#[test]
fn feedback_metrics_allow_open_sectional_table_rect_overlap_as_warning() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let table = test_feedback_placement(
        "white_coffee_table",
        "white coffee table",
        [0.0, 0.0, 0.0],
        [0.30, 0.40, 0.70, 0.70],
    );
    let sofa = test_feedback_placement(
        "tan_open_sectional_sofa",
        "tan open crescent sectional sofa",
        [0.0, 0.0, 0.0],
        [0.13, 0.12, 0.87, 1.0],
    );
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![table, sofa],
        camera: SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.0, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(5.0),
            vertical_fov_degrees: Some(72.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [
            {
                "cache_key": "table",
                "screen_bbox": [0.30, 0.40, 0.70, 0.70],
                "screen_contact": [0.50, 0.55],
                "world_aabb": {
                    "min": [-1.5, 0.0, -0.6],
                    "max": [1.5, 0.4, 0.6]
                }
            },
            {
                "cache_key": "sofa",
                "screen_bbox": [0.13, 0.12, 0.87, 1.0],
                "screen_contact": [0.50, 1.0],
                "world_aabb": {
                    "min": [-1.8, 0.0, -1.2],
                    "max": [1.8, 1.0, 1.2]
                }
            }
        ],
        "camera": { "radius": 5.0 }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/open-sectional.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["projection_passed"], json!(true));
    assert_eq!(metrics["physical_passed"], json!(true));
    assert_eq!(metrics["physical_layout"]["hard_failure_count"], json!(0));
    assert_eq!(
        metrics["physical_layout"]["pairs"][0]["hard_failure"],
        json!(false)
    );
    let deltas = feedback_layout_deltas(&metrics);
    let sofa_delta = json_array3(&deltas["objects"][1]["translation_delta"]).unwrap();
    assert!(sofa_delta[0].abs() + sofa_delta[2].abs() < 0.05);
}

#[test]
fn feedback_selection_score_penalizes_hard_overlap_failures() {
    let clean = json!({
        "score": 0.60,
        "object_count": 2,
        "object_pass_count": 2,
        "rotation_pass_count": 2,
        "projection_passed": true,
        "rotation_passed": true,
        "physical_layout": {
            "hard_failure_count": 0,
            "max_overlap_fraction_smaller": 0.0
        }
    });
    let overlapped = json!({
        "score": 0.95,
        "object_count": 2,
        "object_pass_count": 2,
        "rotation_pass_count": 2,
        "projection_passed": true,
        "rotation_passed": true,
        "physical_layout": {
            "hard_failure_count": 1,
            "max_overlap_fraction_smaller": 1.0
        }
    });

    assert!(feedback_selection_score(&clean) > feedback_selection_score(&overlapped));
}

#[test]
fn feedback_selection_score_prefers_projection_over_rotation_only_candidate() {
    let projection_fixed = json!({
        "score": 0.7625,
        "object_count": 7,
        "object_pass_count": 7,
        "rotation_pass_count": 3,
        "projection_passed": true,
        "rotation_passed": false,
        "physical_layout": {
            "hard_failure_count": 0,
            "max_overlap_fraction_smaller": 0.0
        }
    });
    let rotation_only = json!({
        "score": 0.7197,
        "object_count": 7,
        "object_pass_count": 5,
        "rotation_pass_count": 7,
        "projection_passed": false,
        "rotation_passed": true,
        "physical_layout": {
            "hard_failure_count": 0,
            "max_overlap_fraction_smaller": 0.0
        }
    });

    assert!(feedback_selection_score(&projection_fixed) > feedback_selection_score(&rotation_only));
}

#[test]
fn feedback_selection_score_uses_valid_scene_quality_rubric() {
    let low_rubric = json!({
        "score": 0.70,
        "object_count": 2,
        "object_pass_count": 2,
        "rotation_pass_count": 2,
        "projection_passed": true,
        "rotation_passed": true,
        "physical_layout": {
            "hard_failure_count": 0,
            "max_overlap_fraction_smaller": 0.0
        },
        "scene_quality_rubric": {
            "overall_score": 0.30,
            "blocking_issue_count": 1
        }
    });
    let high_rubric = json!({
        "score": 0.70,
        "object_count": 2,
        "object_pass_count": 2,
        "rotation_pass_count": 2,
        "projection_passed": true,
        "rotation_passed": true,
        "physical_layout": {
            "hard_failure_count": 0,
            "max_overlap_fraction_smaller": 0.0
        },
        "scene_quality_rubric": {
            "overall_score": 0.85,
            "blocking_issue_count": 0
        }
    });

    assert!(feedback_selection_score(&high_rubric) > feedback_selection_score(&low_rubric));
}

#[test]
fn rubric_gate_rejects_blocking_scene_quality_issue() {
    let mut metrics = json!({
        "passed": true,
        "scene_quality_rubric": {
            "overall_score": 0.82,
            "blocking_issue_count": 1
        }
    });

    apply_feedback_scene_quality_rubric_gate(&mut metrics);

    assert_eq!(metrics["passed"], json!(false));
    assert_eq!(metrics["rubric_passed"], json!(false));
}

#[test]
fn feedback_predictive_delta_prevents_move_into_table() {
    let placement = test_feedback_placement(
        "conference_chair",
        "conference chair",
        [0.0, 0.0, -1.0],
        [0.45, 0.7, 0.55, 0.95],
    );
    let footprints = vec![
        Some(FeedbackFootprint {
            index: 0,
            kind: FeedbackPhysicalKind::Table,
            rect: FootprintRect {
                min_x: -1.0,
                min_z: -0.5,
                max_x: 1.0,
                max_z: 0.5,
            },
        }),
        Some(FeedbackFootprint {
            index: 1,
            kind: FeedbackPhysicalKind::Seating,
            rect: FootprintRect {
                min_x: -0.25,
                min_z: -1.2,
                max_x: 0.25,
                max_z: -0.8,
            },
        }),
    ];

    let correction = feedback_predictive_physical_delta(
        1,
        &placement,
        [0.0, 0.0, 0.7],
        &footprints,
        FeedbackThresholdProfile::Standard.thresholds(),
    );

    assert!(correction[2] < -0.2);
}

#[test]
fn feedback_yaw_prefers_canonical_source_pose_over_table_facing() {
    let physical = empty_physical_layout();
    let mut placement = test_feedback_placement(
        "conference_chair",
        "conference chair",
        [1.0, 0.0, 0.0],
        [0.6, 0.5, 0.8, 0.9],
    );
    placement.rotation_y_degrees = 35.0;

    let correction = feedback_yaw_correction(0, &placement, 0.0, &physical);

    assert_eq!(correction.basis, "canonical-bsn-yaw");
    assert!(correction.delta_degrees > 3.0);
}

#[test]
fn feedback_deltas_damp_camera_ray_scale_until_contact_converges() {
    let metrics = json!({
        "objects": [
            {
                "index": 0,
                "object_id": "table",
                "cache_key": "table-cache",
                "expected_bbox": [0.2, 0.5, 0.8, 1.0],
                "observed_bbox": [0.1, 0.2, 0.9, 1.2],
                "translation_delta": [0.0, 0.0, -0.5],
                "grounding_basis": "camera-ray-ground-plane",
                "center_error": 0.24,
                "contact_error": 0.31,
                "scale_multiplier": 0.82,
                "yaw_delta_degrees": 0.0
            }
        ]
    });

    let deltas = feedback_layout_deltas(&metrics);
    let scale = deltas["objects"][0]["scale_multiplier"].as_f64().unwrap();
    let camera_radius = deltas["camera"]["radius_multiplier"].as_f64().unwrap();

    assert!((scale - 1.0).abs() <= 1.0e-6);
    assert!((camera_radius - 1.0).abs() <= 1.0e-6);
}

#[test]
fn feedback_yaw_uses_live_world_item_against_canonical_bsn_yaw() {
    let placement = GroundedScenePlacement {
        entity_id: "chair_1".to_string(),
        asset_id: "chair".to_string(),
        object_id: "chair".to_string(),
        instance_id: Some("chair_1".to_string()),
        label: "chair".to_string(),
        source_bbox: [0.1, 0.2, 0.3, 0.7],
        contact_pixel: [0.2, 0.7],
        ground_point: [0.0, 0.0, 0.0],
        translation: [0.0, 0.0, 0.0],
        rotation_y_degrees: 90.0,
        scale: [1.0, 1.0, 1.0],
        local_aabb: SceneAssetAabb {
            min: [-0.3, 0.0, -0.4],
            max: [0.3, 1.0, 0.4],
        },
        target_footprint_m: [0.6, 0.8],
    };

    let correction = feedback_yaw_correction(0, &placement, 20.0, &empty_physical_layout());

    assert_eq!(correction.basis, "canonical-bsn-yaw");
    assert!(correction.delta_degrees > 20.0);
}

#[test]
fn feedback_rotation_selection_exposes_relative_candidate_choices() {
    let selection = feedback_rotation_selection(170.0, 22.0, "canonical-bsn-yaw");

    assert_eq!(
        selection["search_strategy"],
        json!("bounded-coarse-to-fine-plus-cardinal-yaw")
    );
    assert!(
        selection["instruction"]
            .as_str()
            .unwrap()
            .contains("candidate_index only")
    );
    let candidates = selection["candidates"].as_array().unwrap();
    assert!(candidates.len() >= 7);
    assert!(candidates.iter().any(|candidate| {
        candidate["yaw_delta_degrees"]
            .as_f64()
            .is_some_and(|delta| (delta - 180.0).abs() <= 1.0e-5)
    }));
    assert!(candidates.iter().any(|candidate| {
        candidate["yaw_delta_degrees"]
            .as_f64()
            .is_some_and(|delta| (delta + 180.0).abs() <= 1.0e-5)
    }));
    let selected_index = selection["selected_candidate_index"].as_u64().unwrap() as usize;
    assert_eq!(candidates[selected_index]["selected"], json!(true));
    let selected_delta = selection["selected_yaw_delta_degrees"].as_f64().unwrap();
    assert!((selected_delta - 22.0).abs() <= 1.0e-5);
    let selected_yaw = selection["selected_yaw_degrees"].as_f64().unwrap();
    assert!((selected_yaw + 168.0).abs() <= 1.0e-5);
}

#[test]
fn feedback_rotation_selector_response_updates_only_existing_candidates() {
    let mut metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "label": "chair",
            "current_yaw_degrees": 10.0,
            "yaw_delta_degrees": 0.0,
            "rotation_selection": feedback_rotation_selection(10.0, 18.0, "canonical-bsn-yaw")
        }]
    });
    let response = SceneRotationSelectionResponse {
        objects: vec![burn_synth_scene::SceneRotationSelection {
            index: 0,
            candidate_index: 2,
            confidence: 0.81,
            rationale: "rendered chair back best matches this relative turn".to_string(),
        }],
    };

    let report = apply_feedback_rotation_selection_response(&mut metrics, &response);

    assert_eq!(report["applied_count"], json!(1));
    let selected_delta = metrics["objects"][0]["yaw_delta_degrees"].as_f64().unwrap();
    let candidates = metrics["objects"][0]["rotation_selection"]["candidates"]
        .as_array()
        .unwrap();
    let expected_delta = candidates
        .iter()
        .find(|candidate| candidate["candidate_index"] == json!(2))
        .unwrap()["yaw_delta_degrees"]
        .as_f64()
        .unwrap();
    assert!((selected_delta - expected_delta).abs() <= 1.0e-5);
    assert_eq!(
        metrics["objects"][0]["rotation_selection"]["selection_source"],
        json!("openai_candidate_selector")
    );
    let confidence = metrics["objects"][0]["rotation_selection"]["selector_result"]["confidence"]
        .as_f64()
        .unwrap();
    assert!((confidence - 0.81).abs() <= 1.0e-5);
}

#[test]
fn feedback_rotation_selector_response_ignores_invalid_candidate_indices() {
    let mut metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "label": "chair",
            "current_yaw_degrees": 10.0,
            "yaw_delta_degrees": 0.0,
            "rotation_selection": feedback_rotation_selection(10.0, 18.0, "canonical-bsn-yaw")
        }]
    });
    let response = SceneRotationSelectionResponse {
        objects: vec![burn_synth_scene::SceneRotationSelection {
            index: 0,
            candidate_index: 999,
            confidence: 0.99,
            rationale: "bad candidate".to_string(),
        }],
    };

    let report = apply_feedback_rotation_selection_response(&mut metrics, &response);

    assert_eq!(report["applied_count"], json!(0));
    assert_eq!(metrics["objects"][0]["yaw_delta_degrees"], json!(0.0));
    assert_eq!(
        report["ignored"][0]["reason"],
        json!("candidate_index_not_available")
    );
}

#[test]
fn feedback_rendered_rotation_selector_uses_best_visual_candidate() {
    let mut rotation_selection = feedback_rotation_selection(15.0, 0.0, "canonical-bsn-yaw");
    let candidates = rotation_selection["candidates"].as_array_mut().unwrap();
    for candidate in candidates.iter_mut() {
        candidate["visual_score"] = json!(0.20);
        candidate["rendered_candidate_crop"] = json!("/tmp/candidate.png");
    }
    let flip_index = candidates
        .iter()
        .position(|candidate| {
            candidate["yaw_delta_degrees"]
                .as_f64()
                .is_some_and(|delta| (delta - 180.0).abs() <= 1.0e-5)
        })
        .unwrap();
    candidates[flip_index]["visual_score"] = json!(0.94);
    let mut metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "label": "swivel chair",
            "current_yaw_degrees": 15.0,
            "yaw_delta_degrees": 0.0,
            "rotation_selection": rotation_selection
        }]
    });

    let report = apply_feedback_rendered_rotation_selection(&mut metrics);

    assert_eq!(report["applied_count"], json!(1));
    assert_eq!(
        metrics["objects"][0]["rotation_selection"]["selection_source"],
        json!("rendered_candidate_sweep")
    );
    assert_eq!(
        metrics["objects"][0]["rotation_selection"]["selected_candidate_index"],
        json!(flip_index)
    );
    assert!((metrics["objects"][0]["yaw_delta_degrees"].as_f64().unwrap() - 180.0).abs() <= 1.0e-5);
    assert_eq!(metrics["objects"][0]["max_yaw_delta_degrees"], json!(180.0));
}

#[test]
fn feedback_visual_rotation_deltas_can_apply_cardinal_flips() {
    let commands = vec![json!({
        "type": "spawn_cached",
        "cache_key": "chair-cache",
        "translation": [0.0, 0.0, 0.0],
        "rotation": quat_from_y_degrees(10.0),
        "scale": [1.0, 1.0, 1.0],
    })];
    let deltas = json!({
        "objects": [{
            "translation_delta": [0.0, 0.0, 0.0],
            "scale_multiplier": 1.0,
            "yaw_delta_degrees": 180.0,
            "max_yaw_delta_degrees": 180.0
        }]
    });

    let adjusted = apply_feedback_deltas_to_commands(&commands, &deltas).unwrap();

    let yaw = adjusted[0]["rotation"]
        .as_array()
        .and_then(|_| adjusted[0]["rotation"].clone().as_array().cloned())
        .and_then(|values| {
            let mut out = [0.0; 4];
            for (slot, value) in out.iter_mut().zip(values) {
                *slot = value.as_f64()? as f32;
            }
            Some(out)
        })
        .map(quat_y_degrees)
        .unwrap();
    assert!((normalize_degrees(yaw - -170.0)).abs() <= 1.0e-3);
}

#[test]
fn feedback_rotation_selection_task_includes_candidate_images() {
    let mut rotation_selection = feedback_rotation_selection(0.0, 0.0, "canonical-bsn-yaw");
    rotation_selection["candidates"][0]["rendered_candidate_crop"] =
        json!("/tmp/iter/objects/candidate_00.png");
    rotation_selection["candidates"][0]["rendered_candidate_full_frame"] =
        json!("/tmp/iter/objects/candidate_00_full.png");
    rotation_selection["candidates"][0]["rendered_candidate_screenshot"] =
        json!("/tmp/iter/objects/candidate_00_full.png");
    let metrics = json!({
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "label": "chair",
            "expected_bbox": [0.1, 0.2, 0.3, 0.7],
            "observed_bbox": [0.11, 0.22, 0.31, 0.72],
            "current_yaw_degrees": 0.0,
            "canonical_yaw_degrees": 0.0,
            "rotation_selection": rotation_selection
        }]
    });
    let object_crops = json!({
        "objects": [{
            "index": 0,
            "source_crop": "/tmp/iter/objects/chair_source.png",
            "isolated_render_full_frame": "/tmp/iter/objects/chair_isolated_full.png",
            "isolated_render_bbox": [0.12, 0.21, 0.32, 0.74],
            "rendered_crop": "/tmp/iter/objects/chair_render.png"
        }]
    });

    let task = feedback_rotation_selection_task(&metrics, &object_crops);
    let paths = feedback_rotation_selection_image_paths(&task);

    assert_eq!(task["objects"][0]["purpose"], Value::Null);
    assert_eq!(
        task["objects"][0]["isolated_render_full_frame"],
        json!("/tmp/iter/objects/chair_isolated_full.png")
    );
    assert!(paths.iter().any(|path| path.ends_with("chair_source.png")));
    assert!(
        paths
            .iter()
            .any(|path| path.ends_with("chair_isolated_full.png"))
    );
    assert!(paths.iter().any(|path| path.ends_with("chair_render.png")));
    assert!(paths.iter().any(|path| path.ends_with("candidate_00.png")));
    assert!(
        paths
            .iter()
            .any(|path| path.ends_with("candidate_00_full.png"))
    );
}

#[test]
fn status_world_item_yaw_matches_cache_key_when_order_differs() {
    let status = json!({
        "world_items": [
            {
                "cache_key": "table-cache",
                "rotation": quat_from_y_degrees(0.0),
            },
            {
                "cache_key": "chair-cache",
                "rotation": quat_from_y_degrees(-90.0),
            }
        ]
    });

    let yaw = status_world_item_yaw_degrees(&status, 0, Some("chair-cache")).unwrap();

    assert!((yaw + 90.0).abs() <= 1.0e-5);
}

#[test]
fn feedback_status_prefers_apply_projection_order_when_ready() {
    let apply_ack = json!({
        "status": {
            "sequence": 1,
            "projected_items": [{
                "screen_bbox": [0.1, 0.1, 0.2, 0.2],
                "world_aabb": {
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0]
                }
            }]
        }
    });
    let capture_ack = json!({
        "acknowledgement": {
            "status": {
                "sequence": 2,
                "projected_items": [{
                    "screen_bbox": [0.3, 0.3, 0.4, 0.4]
                }]
            }
        }
    });

    let status = McpServer::feedback_capture_status(&apply_ack, &capture_ack);

    assert_eq!(status["sequence"], json!(1));
    assert_eq!(
        status["projected_items"][0]["screen_bbox"],
        json!([0.1, 0.1, 0.2, 0.2])
    );
}

#[test]
fn feedback_projected_status_readiness_requires_loaded_aabb_projection() {
    let not_ready = json!({
        "projected_items": [{
            "screen_bbox": null,
            "projected_corners": 0,
            "world_aabb": null
        }]
    });
    let ready = json!({
        "projected_items": [{
            "screen_bbox": [0.1, 0.1, 0.4, 0.5],
            "projected_corners": 8,
            "world_aabb": {
                "min": [-1.0, 0.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            }
        }]
    });

    assert!(!McpServer::feedback_status_projected_items_ready(
        &not_ready, 1
    ));
    assert!(McpServer::feedback_status_projected_items_ready(&ready, 1));
    assert!(!McpServer::feedback_status_projected_items_ready(&ready, 2));
}

#[test]
fn feedback_metrics_use_camera_ray_grounding_when_status_has_world_aabb() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![GroundedScenePlacement {
            entity_id: "chair_1".to_string(),
            asset_id: "chair".to_string(),
            object_id: "chair".to_string(),
            instance_id: Some("chair_1".to_string()),
            label: "chair".to_string(),
            source_bbox: [0.16, 0.63, 0.37, 1.0],
            contact_pixel: [0.347, 0.985],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            },
            target_footprint_m: [0.8, 0.8],
        }],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair",
            "screen_bbox": [0.60, 0.42, 0.82, 0.74],
            "screen_contact": [0.70, 0.66],
            "world_aabb": {
                "min": [-0.5, 0.0, -0.5],
                "max": [0.5, 1.0, 0.5]
            },
            "projected_corners": 8,
            "total_corners": 8
        }],
        "camera": {
            "translation": [-0.00000027, 1.9615524, -3.07795],
            "rotation": [0.0000000083, 0.9816272, 0.19080901, -0.0000000429],
            "yaw": std::f32::consts::PI,
            "pitch": 0.38397244,
            "radius": 3.3142834,
            "vertical_fov_degrees": 70.0
        }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/no_screenshot.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();
    let object = &metrics["objects"][0];
    let translation_delta = json_array3(&object["translation_delta"]).unwrap();

    assert_eq!(object["grounding_basis"], json!("camera-ray-ground-plane"));
    assert!(translation_delta[0] > 0.05);
    assert!(translation_delta[2] < -0.5);
    assert_eq!(object["contact_residual_applied"], json!(true));
    assert!(object["target_ground_point"].is_array());
    assert!(object["observed_ground_point"].is_array());
}

#[test]
fn feedback_metrics_allow_contact_aligned_edge_cropped_chair() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/curry.png".to_string(),
        scene_calibration: None,
        objects: vec![burn_synth_scene::SceneObjectSpec {
            id: "chair_left".to_string(),
            label: "conference chair".to_string(),
            aliases: vec!["chair".to_string()],
            bbox: [0.13, 0.12, 0.87, 1.0],
            instances: Vec::new(),
            representative_instance_id: None,
            reuse_group: Some("chair".to_string()),
            instance_count: 1,
            object_prompt: "conference chair".to_string(),
            camera_hint: None,
            rotation_hint_degrees: None,
            target_footprint_m: Some([0.7, 0.7]),
        }],
    };
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![test_feedback_placement(
            "chair_left",
            "conference chair",
            [0.0, 0.0, 0.0],
            [0.13, 0.12, 0.87, 1.0],
        )],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair_left",
            "screen_bbox": [0.16, 0.62, 0.84, 1.74],
            "screen_contact": [0.50, 1.0],
            "projected_corners": 8,
            "total_corners": 8
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/edge-cropped.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["object_pass_count"], json!(1));
    assert_eq!(metrics["objects"][0]["passed"], json!(true));
    assert_eq!(metrics["objects"][0]["source_edge_cropped"], json!(true));
}

#[test]
fn feedback_metrics_reject_bad_edge_cropped_sofa_projection() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/curry.png".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![test_feedback_placement(
            "tan_open_sectional_sofa",
            "tan open crescent sectional sofa",
            [0.0, 0.0, 0.0],
            [0.13, 0.12, 0.87, 1.0],
        )],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "tan_open_sectional_sofa",
            "screen_bbox": [0.23055391, 0.6355549, 1.0096982, 1.9141463],
            "screen_contact": [0.50, 1.0],
            "projected_corners": 8,
            "total_corners": 8
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/bad-curry-sofa.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["object_pass_count"], json!(0));
    assert_eq!(metrics["objects"][0]["passed"], json!(false));
    assert_eq!(metrics["objects"][0]["source_edge_cropped"], json!(true));
    assert!(
        metrics["objects"][0]["bbox_overscan"].as_f64().unwrap()
            > metrics["objects"][0]["max_bbox_overscan"].as_f64().unwrap()
    );
    assert!(metrics["objects"][0]["scale_multiplier"].as_f64().unwrap() < 0.90);
}

#[test]
fn feedback_metrics_use_visible_bbox_for_frame_clipped_seating() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![test_feedback_placement(
            "chair_left_near",
            "conference chair",
            [0.0, 0.0, 0.0],
            [0.162, 0.531, 0.339, 0.972],
        )],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair_left_near",
            "screen_bbox": [0.15, 0.61, 0.35, 1.12],
            "screen_contact": [0.2505, 0.972],
            "projected_corners": 8,
            "total_corners": 8
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/clipped-chair.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["object_pass_count"], json!(1));
    assert_eq!(metrics["projection_passed"], json!(true));
    assert_eq!(metrics["objects"][0]["source_edge_cropped"], json!(true));
    assert_eq!(metrics["objects"][0]["visible_bbox_scoring"], json!(true));
    let visible = json_array4(&metrics["objects"][0]["visible_observed_bbox"]).unwrap();
    assert!((visible[0] - 0.15).abs() <= 1.0e-5);
    assert!((visible[1] - 0.61).abs() <= 1.0e-5);
    assert!((visible[2] - 0.35).abs() <= 1.0e-5);
    assert!((visible[3] - 1.0).abs() <= 1.0e-5);
}

#[test]
fn feedback_metrics_do_not_fail_reused_chair_for_area_only_when_contact_aligned() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let mut placement = test_feedback_placement(
        "chair_group",
        "reusable conference chair group",
        [0.0, 0.0, 0.0],
        [0.495, 0.045, 0.645, 0.400],
    );
    placement.instance_id = Some("chair_center".to_string());
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![placement],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair_group",
            "screen_bbox": [0.530, 0.168, 0.641, 0.427],
            "screen_contact": [0.570, 0.385],
            "projected_corners": 8,
            "total_corners": 8
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/reused-chair.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["object_pass_count"], json!(1));
    assert_eq!(metrics["objects"][0]["passed"], json!(true));
    assert!(metrics["objects"][0]["area_log2_error"].as_f64().unwrap() > 0.65);
}

#[test]
fn feedback_metrics_require_yaw_convergence_before_projection_passes() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.png".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: String::new(),
        placements: vec![test_feedback_placement(
            "chair",
            "conference chair",
            [0.0, 0.0, 0.0],
            [0.40, 0.40, 0.60, 0.80],
        )],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair",
            "screen_bbox": [0.40, 0.40, 0.60, 0.80],
            "screen_contact": [0.50, 0.80],
            "projected_corners": 8,
            "total_corners": 8
        }],
        "world_items": [{
            "cache_key": "chair",
            "rotation": quat_from_y_degrees(30.0)
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/yaw.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();

    assert_eq!(metrics["objects"][0]["yaw_passed"], json!(false));
    assert_eq!(metrics["objects"][0]["passed"], json!(true));
    assert_eq!(metrics["projection_passed"], json!(true));
    assert_eq!(metrics["rotation_passed"], json!(false));
    assert_eq!(metrics["passed"], json!(false));
    assert!(
        metrics["objects"][0]["yaw_delta_abs_degrees"]
            .as_f64()
            .unwrap()
            > 8.0
    );
    let deltas = feedback_layout_deltas(&metrics);
    assert!(deltas["objects"][0]["yaw_delta_degrees"].as_f64().unwrap() < -8.0);
}

#[test]
fn feedback_metrics_use_bbox_center_anchor_for_tabletops() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![GroundedScenePlacement {
            entity_id: "table_1".to_string(),
            asset_id: "table".to_string(),
            object_id: "conference_table".to_string(),
            instance_id: None,
            label: "conference table".to_string(),
            source_bbox: [0.4, 0.5, 0.6, 1.0],
            contact_pixel: [0.5, 1.0],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-1.0, 0.0, -0.4],
                max: [1.0, 0.2, 0.4],
            },
            target_footprint_m: [2.0, 0.8],
        }],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "table",
            "screen_bbox": [0.4, 0.3, 0.6, 0.7],
            "screen_contact": [0.5, 0.45],
            "world_aabb": {
                "min": [-1.0, 0.0, -0.4],
                "max": [1.0, 0.2, 0.4]
            },
            "projected_corners": 8,
            "total_corners": 8
        }],
        "camera": {
            "translation": [0.0, 2.0, -3.0],
            "rotation": [0.0, 0.9816272, 0.19080901, 0.0],
            "yaw": std::f32::consts::PI,
            "pitch": 0.38397244,
            "radius": 3.3142834,
            "vertical_fov_degrees": 70.0
        }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/no_screenshot.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();
    let object = &metrics["objects"][0];
    let translation_delta = json_array3(&object["translation_delta"]).unwrap();

    assert_eq!(object["anchor_basis"], json!("bbox-center"));
    assert_eq!(object["expected_anchor"], json!([0.5, 0.75]));
    assert_eq!(object["observed_anchor"], json!([0.5, 0.5]));
    assert!(translation_delta[2].abs() <= 0.850001);
}

#[test]
fn feedback_metrics_relax_centered_edge_cropped_table_area() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![GroundedScenePlacement {
            entity_id: "table_1".to_string(),
            asset_id: "table".to_string(),
            object_id: "conference_table".to_string(),
            instance_id: None,
            label: "conference table".to_string(),
            source_bbox: [0.299, 0.475, 0.648, 1.0],
            contact_pixel: [0.4735, 1.0],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-1.0, 0.0, -0.4],
                max: [1.0, 0.2, 0.4],
            },
            target_footprint_m: [4.2, 1.4],
        }],
        camera: SceneCamera {
            translation: [0.0, 2.0, -3.0],
            focus: [0.0, 0.7, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(4.0),
            vertical_fov_degrees: Some(70.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "table",
            "screen_bbox": [0.402, 0.518, 0.526, 0.963],
            "screen_contact": [0.474, 0.738],
            "world_aabb": {
                "min": [-1.0, 0.0, -0.4],
                "max": [1.0, 0.2, 0.4]
            },
            "projected_corners": 8,
            "total_corners": 8
        }]
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/no_screenshot.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();
    let object = &metrics["objects"][0];

    assert_eq!(object["source_edge_cropped"], json!(true));
    assert_eq!(object["anchor_basis"], json!("bbox-center"));
    assert!(object["area_log2_error"].as_f64().unwrap() > 1.0);
    assert_eq!(object["passed"], json!(true));
    assert_eq!(metrics["projection_passed"], json!(true));
}

#[test]
fn feedback_deltas_do_not_axis_scale_edge_cropped_tables() {
    let metrics = json!({
        "score": 0.61,
        "thresholds": {
            "max_center_error": 0.08,
            "max_contact_error": 0.08,
            "max_area_log2_error": 0.65,
            "min_overall_score": 0.55,
            "max_seating_table_overlap_fraction": 0.38,
            "max_seating_table_penetration_m": 0.18,
            "max_seating_seating_overlap_fraction": 0.42,
            "max_seating_seating_penetration_m": 0.12
        },
        "objects": [{
            "index": 0,
            "object_id": "white_conference_table_01",
            "label": "white rectangular conference table",
            "cache_key": "white_conference_table_01_asset",
            "expected_bbox": [0.299, 0.475, 0.648, 1.0],
            "observed_bbox": [0.3107, 0.4609, 0.5918, 1.1847],
            "visible_bbox_scoring": true,
            "source_edge_cropped": true,
            "grounding_basis": "camera-ray-ground-plane",
            "contact_error": 0.088,
            "center_error": 0.023,
            "area_log2_error": 0.274,
            "translation_delta": [0.03, 0.0, -0.14],
            "scale_multiplier": 0.95,
            "yaw_delta_degrees": 0.0,
            "physical_kind": "table",
            "physical_failures": [],
            "world_footprint": {
                "min_x": -1.0,
                "min_z": -0.5,
                "max_x": 1.0,
                "max_z": 0.5
            }
        }],
        "physical_layout": {
            "pairs": []
        }
    });

    let deltas = feedback_layout_deltas(&metrics);
    assert!(deltas["objects"][0]["scale_multiplier_xyz"].is_null());
    assert_eq!(deltas["objects"][0]["scale_multiplier"], json!(1.0));
    assert_eq!(
        deltas["objects"][0]["scale_source"],
        json!("object_projection")
    );
}

#[test]
fn feedback_deltas_apply_full_bounded_scale_for_large_edge_cropped_sofa() {
    let metrics = json!({
        "score": 0.70,
        "thresholds": {
            "max_center_error": 0.08,
            "max_contact_error": 0.08,
            "max_area_log2_error": 0.65,
            "min_overall_score": 0.55,
            "max_seating_table_overlap_fraction": 0.38,
            "max_seating_table_penetration_m": 0.18,
            "max_seating_seating_overlap_fraction": 0.42,
            "max_seating_seating_penetration_m": 0.12
        },
        "objects": [{
            "index": 0,
            "object_id": "tan_open_sectional_sofa",
            "label": "tan open crescent sectional sofa",
            "cache_key": "tan_open_sectional_sofa_asset",
            "expected_bbox": [0.13, 0.12, 0.871, 1.0],
            "observed_bbox": [0.209, 0.654, 0.793, 1.628],
            "source_edge_cropped": true,
            "visible_bbox_scoring": true,
            "grounding_basis": "camera-ray-ground-plane",
            "contact_error": 0.0047,
            "center_error": 0.267,
            "area_log2_error": 1.69,
            "translation_delta": [0.0, 0.0, 0.16],
            "scale_multiplier": 1.071,
            "yaw_delta_degrees": 0.0,
            "physical_kind": "other",
            "physical_failures": [],
            "world_footprint": {
                "min_x": -1.0,
                "min_z": -0.5,
                "max_x": 1.0,
                "max_z": 0.5
            }
        }],
        "physical_layout": {
            "pairs": []
        }
    });

    let deltas = feedback_layout_deltas(&metrics);
    assert!(deltas["objects"][0]["scale_multiplier"].as_f64().unwrap() > 1.06);
    assert_eq!(
        deltas["objects"][0]["scale_source"],
        json!("object_projection")
    );
}

#[test]
fn feedback_metrics_emit_bounded_corrections_for_projection_mismatch() {
    let manifest = SceneObjectManifest {
        source_scene_path: "/tmp/source.jpg".to_string(),
        scene_calibration: None,
        objects: Vec::new(),
    };
    let layout = GroundedSceneLayout {
        bsn: "scene {}".to_string(),
        placements: vec![GroundedScenePlacement {
            entity_id: "chair_1".to_string(),
            asset_id: "chair".to_string(),
            object_id: "chair".to_string(),
            instance_id: Some("chair_1".to_string()),
            label: "chair".to_string(),
            source_bbox: [0.4, 0.4, 0.6, 0.7],
            contact_pixel: [0.5, 0.7],
            ground_point: [0.0, 0.0, 0.0],
            translation: [0.0, 0.0, 0.0],
            rotation_y_degrees: 0.0,
            scale: [1.0, 1.0, 1.0],
            local_aabb: SceneAssetAabb {
                min: [-0.5, 0.0, -0.5],
                max: [0.5, 1.0, 0.5],
            },
            target_footprint_m: [0.8, 0.8],
        }],
        camera: SceneCamera {
            translation: [0.0, 2.0, 5.0],
            focus: [0.0, 0.0, 0.0],
            yaw: Some(180.0),
            pitch: Some(25.0),
            radius: Some(5.0),
            vertical_fov_degrees: Some(72.0),
        },
        rug_center: [0.0, 0.0, 0.0],
        rug_scale: [1.0, 1.0, 1.0],
        projection_fit: None,
    };
    let status = json!({
        "projected_items": [{
            "cache_key": "chair",
            "screen_bbox": [0.45, 0.5, 0.55, 0.6],
            "screen_contact": [0.5, 0.6],
            "projected_corners": 8,
            "total_corners": 8
        }],
        "camera": {
            "radius": 5.0
        }
    });

    let metrics = scene_feedback_metrics(
        &manifest,
        &layout,
        &status,
        Path::new("/tmp/iter.png"),
        FeedbackThresholdProfile::Standard.thresholds(),
        FeedbackThresholdProfile::Standard,
    )
    .unwrap();
    let deltas = feedback_layout_deltas(&metrics);

    assert!(!metrics["passed"].as_bool().unwrap());
    assert_eq!(metrics["object_count"], json!(1));
    assert_eq!(metrics["object_pass_count"], json!(0));
    let object_delta = &deltas["objects"][0];
    let translation_delta = object_delta["translation_delta"].as_array().unwrap();
    assert!(translation_delta[0].as_f64().unwrap().abs() <= 1.0e-6);
    assert!(translation_delta[1].as_f64().unwrap().abs() <= 1.0e-6);
    assert!((translation_delta[2].as_f64().unwrap() - 0.2).abs() <= 1.0e-5);
    let scale = object_delta["scale_multiplier"].as_f64().unwrap();
    assert!((scale - 1.22).abs() <= 1.0e-5);
    let yaw_delta = object_delta["yaw_delta_degrees"].as_f64().unwrap();
    assert!(yaw_delta.abs() <= 1.0e-6);
    assert_eq!(
        metrics["objects"][0]["yaw_basis"],
        json!("canonical-bsn-yaw-within-threshold")
    );
}

#[test]
fn apply_mesh_decimation_preserves_pbr_baked_meshes() {
    let mesh = Mesh {
        vertices: vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        faces: vec![[0, 1, 2], [1, 3, 2]],
        uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        normals: Vec::new(),
        material: None,
        pbr_textures: Some(burn_synth::MeshPbrTextures {
            base_color: burn_synth::MeshTexture {
                width: 1,
                height: 1,
                rgba8: vec![255, 255, 255, 255],
            },
            metallic_roughness: burn_synth::MeshTexture {
                width: 1,
                height: 1,
                rgba8: vec![0, 255, 0, 255],
            },
            normal: None,
            emissive: None,
            occlusion: None,
        }),
    };

    let output = apply_mesh_decimation(mesh.clone(), Some(1)).expect("decimation");

    assert_eq!(output.faces.len(), mesh.faces.len());
    assert!(output.pbr_textures.is_some());
}

#[test]
fn scene_compose_plan_generates_spawn_commands_with_validation_keys() {
    let plan = compose_scene_layout(SceneComposeArgs {
        reference_objects: vec![scene_layout::SceneReferenceObject {
            id: Some("chair_1".to_string()),
            label: "chair".to_string(),
            aliases: Vec::new(),
            bbox: [0.1, 0.2, 0.3, 0.6],
        }],
        assets: vec![scene_layout::SceneAssetBinding {
            reference_id: Some("chair_1".to_string()),
            label: Some("chair".to_string()),
            aliases: Vec::new(),
            path: Some(PathBuf::from("/tmp/chair.glb")),
            cache_key: None,
            local_aabb: None,
            select: true,
        }],
        apply: false,
        clear_existing: true,
        layout_width: 6.0,
        layout_depth: 4.0,
        y: 0.0,
        min_scale: 0.35,
        scale_multiplier: 1.0,
    })
    .expect("compose plan");
    let commands = scene_commands_from_plan(&plan).expect("scene commands");
    assert_eq!(commands.len(), 2);
    assert_eq!(commands[0]["type"], "clear_scene");
    assert_eq!(commands[1]["type"], "spawn_path");
    assert_eq!(commands[1]["cache_key"], "path:/tmp/chair.glb");
    assert_eq!(commands[1]["select"], true);
}

#[test]
fn composition_candidate_modes_compare_cv_grounded_against_heuristic_only_with_feedback() {
    assert_eq!(
        scene_composition_candidate_modes(SceneCompositionMode::CvGrounded, true),
        vec![
            SceneCompositionMode::CvGrounded,
            SceneCompositionMode::Heuristic
        ]
    );
    assert_eq!(
        scene_composition_candidate_modes(SceneCompositionMode::CvGrounded, false),
        vec![SceneCompositionMode::CvGrounded]
    );
    assert_eq!(
        scene_composition_candidate_modes(SceneCompositionMode::Heuristic, true),
        vec![SceneCompositionMode::Heuristic]
    );
}

#[test]
fn feedback_result_selection_score_prefers_accepted_candidate() {
    let accepted = json!({
        "accepted": true,
        "best_score": 0.62,
    });
    let unaccepted = json!({
        "accepted": false,
        "best_score": 0.99,
    });

    assert!(
        feedback_result_selection_score(&accepted) > feedback_result_selection_score(&unaccepted)
    );
}

#[test]
fn feedback_result_selection_score_uses_final_evidence() {
    let best_iteration_only = json!({
        "accepted": false,
        "best_score": 0.42,
    });
    let final_improved = json!({
        "accepted": false,
        "best_score": 0.42,
        "final_evidence": {
            "metrics": {
                "score": 0.70,
                "object_count": 2,
                "object_pass_count": 2,
                "rotation_pass_count": 2,
                "projection_passed": true,
                "rotation_passed": true,
                "physical_layout": {
                    "hard_failure_count": 0,
                    "max_overlap_fraction_smaller": 0.0
                }
            }
        }
    });
    let final_accepted = json!({
        "accepted": true,
        "best_score": 0.42,
        "final_evidence": {
            "metrics": {
                "score": 0.70,
                "object_count": 2,
                "object_pass_count": 2,
                "rotation_pass_count": 2,
                "projection_passed": true,
                "rotation_passed": true,
                "physical_layout": {
                    "hard_failure_count": 0,
                    "max_overlap_fraction_smaller": 0.0
                }
            }
        }
    });

    assert!(
        feedback_result_selection_score(&final_improved)
            > feedback_result_selection_score(&best_iteration_only)
    );
    assert!(
        feedback_result_selection_score(&final_accepted)
            > feedback_result_selection_score(&final_improved)
    );
}

#[test]
fn feedback_iteration_context_links_screenshots_and_transform_deltas() {
    let previous_commands = vec![
        json!({"type": "clear_scene"}),
        json!({
            "type": "spawn_cached",
            "cache_key": "chair",
            "translation": [0.0, 0.0, 0.0],
            "rotation": quat_from_y_degrees(10.0),
            "scale": [1.0, 1.0, 1.0],
        }),
        json!({"type": "set_camera", "radius": 5.0}),
    ];
    let current_commands = vec![
        json!({"type": "clear_scene"}),
        json!({
            "type": "spawn_cached",
            "cache_key": "chair",
            "translation": [0.25, 0.0, -0.5],
            "rotation": quat_from_y_degrees(25.0),
            "scale": [1.2, 1.2, 1.2],
        }),
        json!({"type": "set_camera", "radius": 4.0}),
    ];
    let previous_iteration = json!({
        "iteration": 0,
        "screenshot": "/tmp/iter_00/screenshot.png",
        "metrics": { "passed": false, "score": 0.4 },
        "layout_delta": { "objects": [] },
    });
    let metrics = json!({
        "passed": false,
        "score": 0.6,
        "projection_passed": false,
        "physical_passed": true,
        "object_count": 1,
        "object_pass_count": 0,
        "physical_layout": { "hard_failure_count": 0 },
        "objects": [{
            "index": 0,
            "object_id": "chair",
            "instance_id": "chair_1",
            "label": "chair",
            "expected_bbox": [0.10, 0.20, 0.30, 0.70],
            "observed_bbox": [0.12, 0.22, 0.32, 0.72],
            "current_yaw_degrees": 25.0,
            "canonical_yaw_degrees": 10.0,
            "rotation_selection": feedback_rotation_selection(25.0, -5.0, "canonical-bsn-yaw")
        }],
    });
    let layout_delta = json!({
        "objects": [{
            "translation_delta": [0.1, 0.0, 0.2],
            "scale_multiplier": 1.1,
            "yaw_delta_degrees": -5.0
        }]
    });
    let object_crops = json!({
        "objects": [{
            "index": 0,
            "source_crop": "/tmp/iter_01/objects/00_chair_source.png",
            "rendered_crop": "/tmp/iter_01/objects/00_chair_render.png"
        }]
    });

    let context = feedback_iteration_context(
        1,
        Some(&previous_iteration),
        Some(&previous_commands),
        &current_commands,
        Path::new("/tmp/iter_01/screenshot.png"),
        &metrics,
        &layout_delta,
        &object_crops,
    );

    assert_eq!(
        context["previous_iteration"]["screenshot"],
        json!("/tmp/iter_00/screenshot.png")
    );
    assert_eq!(
        context["current_iteration"]["screenshot"],
        json!("/tmp/iter_01/screenshot.png")
    );
    assert_eq!(
        context["current_iteration"]["object_crops"]["objects"][0]["source_crop"],
        json!("/tmp/iter_01/objects/00_chair_source.png")
    );
    assert_eq!(
        context["current_iteration"]["rotation_selection_task"]["purpose"],
        json!("bounded-object-rotation-selection")
    );
    assert_eq!(
        context["current_iteration"]["rotation_selection_task"]["objects"][0]["source_crop"],
        json!("/tmp/iter_01/objects/00_chair_source.png")
    );
    assert!(
        context["current_iteration"]["rotation_selection_task"]["objects"][0]["rotation_selection"]
            ["candidates"]
            .as_array()
            .unwrap()
            .len()
            >= 5
    );
    assert_eq!(
        context["command_transform_delta_from_previous"]["objects"][0]["translation_delta"],
        json!([0.25, 0.0, -0.5])
    );
    let yaw_delta =
        context["command_transform_delta_from_previous"]["objects"][0]["yaw_delta_degrees"]
            .as_f64()
            .unwrap();
    assert!((yaw_delta - 15.0).abs() < 1.0e-3);
    assert_eq!(
        context["current_iteration"]["transform_delta_to_next"],
        layout_delta
    );
}

#[test]
fn scene_sequence_is_strictly_monotonic() {
    let first = next_scene_sequence();
    let second = next_scene_sequence();
    assert!(second > first);
}

#[test]
fn scene_build_candidate_policy_batches_quality_candidates_for_mesh_fallback() {
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "candidate_count": 3
    }))
    .expect("scene build args deserialize");

    let policy = scene_object_image_generation_policy(&args, 2);

    assert_eq!(policy.max_attempts_per_object, 2);
    assert_eq!(policy.candidates_per_attempt, 2);
    assert_eq!(policy.min_score, DEFAULT_SCENE_RECONSTRUCTION_IMAGE_SCORE);

    let draft: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "candidate_count": 3,
        "quality_profile": "draft"
    }))
    .expect("scene build args deserialize");
    let draft_policy = scene_object_image_generation_policy(&draft, 2);
    assert_eq!(draft_policy.max_attempts_per_object, 3);
    assert_eq!(draft_policy.candidates_per_attempt, 1);

    let explicit: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "candidate_count": 3,
        "candidate_batch_size": 2,
        "candidate_retry_attempts": 5,
        "min_reconstruction_score": 0.72,
        "segmentation_provider": "bbox-prompt",
        "segmentation_precision": "bf16",
        "segmentation_quantization": "q8"
    }))
    .expect("scene build args deserialize");
    let policy = scene_object_image_generation_policy(&explicit, 2);
    assert_eq!(policy.max_attempts_per_object, 5);
    assert_eq!(policy.candidates_per_attempt, 2);
    assert_eq!(policy.min_score, 0.72);
    assert_eq!(
        explicit.segmentation_provider,
        Some(SceneSegmentationProvider::BboxPrompt)
    );
    assert_eq!(
        explicit.segmentation_precision,
        Some(SceneSegmentationPrecision::Bf16)
    );
    assert_eq!(
        explicit.segmentation_quantization,
        Some(SceneSegmentationQuantization::Q8)
    );
}

#[test]
fn scene_build_defaults_select_bare_bones_geometric_placement_pipeline() {
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg"
    }))
    .expect("scene build args deserialize");
    let config = ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]));
    let segmentation_provider = args
        .segmentation_provider
        .unwrap_or(config.scene_segmentation_provider);

    let plan = scene_placement_pipeline_plan(ScenePlacementPipelineSelection {
        entry_point: ScenePlacementEntryPoint::SceneBuild,
        lift_assets: args.lift_assets,
        composition_mode: args.composition_mode,
        pose_fit: args.pose_fit,
        canonical_pose: args.canonical_pose,
        scale_policy: args.scale_policy,
        ground_calibration: args.ground_calibration,
        instance_generation: args.instance_generation,
        depth_provider: args.depth_provider,
        locator: args.locator,
        segmentation_provider,
        feedback: args.feedback,
        feedback_iters: args.feedback_iters,
        feedback_rotation_selector: args.feedback_rotation_selector,
        feedback_rubric_scorer: args.feedback_rubric_scorer,
        rotation_fit: args.rotation_fit,
        object_pose_refinement: args.object_pose_refinement,
        object_pose_refinement_set: args.object_pose_refinement_set,
        max_pose_candidates: args.max_pose_candidates,
    });

    assert_eq!(plan.quality_profile, "bare_bones_geometric");
    assert!(plan.warnings.is_empty(), "{:#?}", plan.warnings);
    assert_eq!(
        plan.active_pose_optimizer,
        "visible_surface_dense_depth_search_plus_soft_point_refinement_plus_object_refinement"
    );
    assert!(plan.stages.iter().any(|stage| {
        stage.stage == "object_discretization"
            && stage.enabled
            && stage.method == "locate_anything_burn_native"
    }));
    assert!(plan.stages.iter().any(|stage| {
        stage.stage == "mask_selection" && stage.enabled && stage.method == "sam2"
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
    assert_eq!(args.scale_policy, SceneScalePolicy::AssetPreserving);
    assert!(!args.feedback);
    assert_eq!(args.rotation_fit, SceneRotationFitMode::Off);
    assert_eq!(
        args.object_pose_refinement,
        SceneObjectPoseRefinementMode::GatedGpt
    );
    assert_eq!(
        args.object_pose_refinement_set,
        SceneObjectPoseRefinementSet::TablesAndLargeSeating
    );
}

#[test]
fn scene_asset_lift_policy_rejects_triposplat_for_mesh_pose_fit() {
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "synthesis_models": ["triposplat"],
        "composition_mode": "cv-grounded",
        "pose_fit": "rendered-silhouette",
        "object_pose_refinement": "gated-gpt",
        "object_pose_refinement_set": "tables-and-large-seating"
    }))
    .expect("scene build args deserialize");

    let err = scene_asset_lift_policy(&args).expect_err("splat should not satisfy mesh pose fit");
    assert!(
        err.contains("requires GLB mesh assets"),
        "unexpected error: {err}"
    );

    let mut projected_only: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "synthesis_models": ["triposplat"],
        "composition_mode": "cv-grounded",
        "pose_fit": "projected-aabb",
        "object_pose_refinement": "off"
    }))
    .expect("scene build args deserialize");
    projected_only.rotation_fit = SceneRotationFitMode::Off;
    let policy = scene_asset_lift_policy(&projected_only).expect("projected-only splat ablation");
    assert_eq!(policy.output_format, AssetOutputFormat::Auto);
    assert!(!policy.requires_mesh_assets);
}

#[test]
fn scene_asset_contract_rejects_splat_outputs_for_mesh_scene_fit() {
    let policy = SceneAssetLiftPolicy {
        synthesis_models: vec![SynthesisModel::Trellis],
        output_format: AssetOutputFormat::Glb,
        requires_mesh_assets: true,
        warnings: Vec::new(),
    };
    let outputs = json!({
        "items": [
            {
                "id": "asset_0",
                "input_image_path": "/tmp/table.png",
                "asset_kind": "gaussian_splat",
                "output_format": "splat",
                "synthesis_backend": "triposplat",
                "local_aabb": null
            }
        ]
    });

    let err = validate_scene_asset_outputs_for_policy(&outputs, &policy)
        .expect_err("splat output must not pass mesh scene contract");
    assert!(err.contains("asset_kind=gaussian_splat"), "{err}");
    assert!(err.contains("output_format=splat"), "{err}");
    assert!(err.contains("no local_aabb"), "{err}");
}

#[test]
fn scene_asset_lift_chunk_size_defaults_to_per_object_for_review_runs() {
    assert_eq!(scene_asset_lift_chunk_size(None, 4), 1);
    assert_eq!(scene_asset_lift_chunk_size(Some(0), 4), 1);
    assert_eq!(scene_asset_lift_chunk_size(Some(2), 4), 2);
    assert_eq!(scene_asset_lift_chunk_size(Some(8), 4), 4);
}

#[test]
fn scene_build_args_accept_asset_synthesis_model_override() {
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": "/tmp/scene.jpg",
        "synthesis_models": ["triposplat", "trellis"]
    }))
    .expect("scene build args deserialize");

    assert_eq!(
        args.synthesis_models,
        Some(vec![SynthesisModel::Triposplat, SynthesisModel::Trellis])
    );
}

#[test]
fn scene_build_progress_events_are_emitted_and_persisted_on_failure() {
    let root = unique_test_dir("scene_build_progress");
    let _ = fs::remove_dir_all(&root);
    let source = root.join("missing_scene.jpg");
    let output_dir = root.join("run");
    let args: SceneBuildFromImageArgs = serde_json::from_value(json!({
        "source_scene_path": source,
        "output_dir": output_dir,
        "write_artifacts": true,
        "lift_assets": false,
        "feedback": false
    }))
    .expect("scene build args deserialize");
    let config = ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]));
    let mut events = Vec::new();

    let result = run_scene_build_from_image_with_progress(config, args, |event| {
        events.push(event);
    });

    assert!(
        result.is_err(),
        "missing source image should fail before model work"
    );
    assert!(
        events
            .iter()
            .any(|event| event.stage == "prepare_openai_inputs"
                && event.phase == SceneBuildProgressPhase::Started),
        "prepare event should be emitted"
    );
    assert!(
        events
            .iter()
            .any(|event| event.stage == "scene_build"
                && event.phase == SceneBuildProgressPhase::Failed),
        "failed event should be emitted"
    );
    let progress_path = root.join("run").join("progress_events.jsonl");
    let progress_log =
        fs::read_to_string(&progress_path).expect("progress jsonl should be written");
    assert!(progress_log.contains("\"stage\":\"scene_build\""));
    assert!(progress_log.contains("\"phase\":\"failed\""));
    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn scene_command_waits_for_matching_status_sequence() {
    let root = unique_test_dir("scene_bridge");
    fs::create_dir_all(&root).expect("create temp dir");
    let command_path = root.join("scene_commands.json");
    let status_path = command_path.with_extension("status.json");
    let config = ServerConfig {
        scene_control_path: Some(command_path.clone()),
        scene_status_path: Some(status_path.clone()),
        scene_timeout: Duration::from_secs(1),
        ..ServerConfig::from_args(ServerArgs::parse_from(["burn_synth_mcp"]))
    };
    let server = McpServer::new(config);
    let status_path_for_thread = status_path.clone();
    let command_path_for_thread = command_path.clone();
    let handle = thread::spawn(move || {
        let started = Instant::now();
        loop {
            if command_path_for_thread.exists() {
                let command =
                    read_scene_status(&command_path_for_thread).expect("command JSON should parse");
                let sequence = command["sequence"].as_u64().expect("sequence");
                atomic_write_json(
                    &status_path_for_thread,
                    &json!({
                        "last_sequence": sequence,
                        "ok": true,
                        "cache_entries": [],
                        "world_items": [],
                        "camera": null,
                        "screenshots": [],
                    }),
                )
                .expect("write status");
                return;
            }
            assert!(started.elapsed() < Duration::from_secs(1));
            thread::sleep(Duration::from_millis(10));
        }
    });

    let response = server
        .send_scene_commands(vec![json!({ "type": "clear_selection" })])
        .expect("scene command should be acknowledged");
    handle.join().expect("status writer thread");
    assert_eq!(response["acknowledged"], true);
    assert!(response["status"]["last_sequence"].as_u64().is_some());
    fs::remove_dir_all(root).expect("remove temp dir");
}

#[test]
fn scene_grounding_report_manifest_uses_expected_counts() {
    let manifest = scene_grounding_report_manifest(
        Path::new("/tmp/source.jpg"),
        &["chair".to_string(), "table".to_string()],
        &["chair=6".to_string()],
    )
    .expect("manifest");

    let chair = manifest
        .objects
        .iter()
        .find(|object| object.id == "chair")
        .expect("chair object");
    let table = manifest
        .objects
        .iter()
        .find(|object| object.id == "table")
        .expect("table object");
    assert_eq!(chair.instance_count, 6);
    assert_eq!(table.instance_count, 1);
}

#[test]
fn scene_grounding_report_quality_flags_count_edge_and_tiny_boxes() {
    let manifest = scene_grounding_report_manifest(
        Path::new("/tmp/source.jpg"),
        &["chair".to_string(), "plant".to_string()],
        &["chair=2".to_string()],
    )
    .expect("manifest");
    let evidence = SceneGroundingEvidence {
        source_image_path: "/tmp/source.jpg".to_string(),
        depth: None,
        segmentation: None,
        detections: Vec::new(),
        camera: burn_synth_scene::EstimatedCamera::default(),
        floor: burn_synth_scene::EstimatedFloorPlane::default(),
        objects: vec![
            report_test_object("chair", Some("one"), [0.10, 0.10, 0.20, 0.20]),
            report_test_object("chair", Some("two"), [0.21, 0.10, 0.23, 0.20]),
            report_test_object("chair", Some("three"), [0.24, 0.10, 0.26, 0.20]),
            report_test_object("plant", None, [0.00, 0.20, 0.10, 0.80]),
        ],
    };

    let quality = scene_grounding_quality_report(&manifest, &evidence, 0.5, 0.5);
    assert_eq!(quality["status"], "warn");
    assert_eq!(quality["warning_count"].as_u64().unwrap(), 4);
    assert!(
        quality["group_warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning["kind"] == "expected_instance_count_mismatch")
    );
    let object_warnings = quality["objects"].as_array().unwrap();
    assert!(object_warnings.iter().any(|object| {
        object["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning.as_str().unwrap().starts_with("bbox_area_tiny"))
    }));
    assert!(object_warnings.iter().any(|object| {
        object["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|warning| warning == "bbox_touches_horizontal_image_edge")
    }));
}

fn report_test_object(
    object_id: &str,
    instance_id: Option<&str>,
    bbox: [f32; 4],
) -> burn_synth_scene::ObjectGroundingEvidence {
    let bbox_area = (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0);
    burn_synth_scene::ObjectGroundingEvidence {
        object_id: object_id.to_string(),
        instance_id: instance_id.map(ToOwned::to_owned),
        reuse_group: Some(object_id.to_string()),
        detection: Some(burn_synth_scene::Detection {
            label: object_id.to_string(),
            bbox,
            point: None,
            confidence: None,
            source_query: object_id.to_string(),
        }),
        mask: Some(burn_synth_scene::ObjectMaskEvidence {
            provider: "test".to_string(),
            model: "test".to_string(),
            bbox,
            score: 1.0,
            area_px: 1,
            image_size: [100, 100],
            mask_rle: Vec::new(),
            center_pixel: None,
            contact_pixel: None,
            coverage: Some(bbox_area * 0.5),
            artifact_path: None,
            mask_png_path: None,
        }),
        asset_id: None,
        contact_pixel: None,
        depth_stats: None,
        candidate_floor_contact_rays: Vec::new(),
        metric_contact_point_m: None,
        target_footprint_m: None,
        provenance: Vec::new(),
    }
}

fn unique_test_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "burn_synth_mcp_{label}_{}_{}",
        std::process::id(),
        nanos
    ))
}

fn find_repo_root_for_test() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("crates/burn_synth_mcp").exists() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

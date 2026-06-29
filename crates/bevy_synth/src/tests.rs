use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::ecs::message::MessageWriter;
use bevy::prelude::*;
use bevy_gaussian_splatting::gaussian::settings::GaussianColorSpace;
use bevy_gaussian_splatting::sort::SortMode;
use bevy_gaussian_splatting::{CloudSettings, PlanarGaussian3d, PlanarGaussian3dHandle};
use bevy_mesh::Mesh as BevyMesh;
use bevy_synth_ui::{
    BurnSynthUiPlugin, BurnSynthUiSystemSet, CatalogDeleteRequest, CatalogState, CatalogStatus,
    ScenePipelineUiSettings, SceneProcessingState, SceneQualityProfileSetting,
    ViewerAabbOverlayMode,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_synth_mcp::{
    SceneBuildExecutionKind, SceneBuildProgressEvent, SceneBuildProgressPhase,
    SceneCanonicalPoseMode, SceneCompositionMode, SceneDepthProvider, SceneLocatorProvider,
    SceneObjectPoseRefinementMode, SceneObjectPoseRefinementSet, ScenePoseFitMode,
    SceneScalePolicy, SceneSegmentationProvider, SynthesisModel as McpSynthesisModel,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_synth_scene::SceneQualityProfile;

#[cfg(not(target_arch = "wasm32"))]
use crate::app::prepare_startup_bsn_scene;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use crate::app::should_share_wgpu_inference_device_for_platform;
use crate::app::{
    CachedMeshInstance, MeshCacheResource, SceneInteractionLock, SceneReadOnlyMode,
    UiVisibilityState, apply_scene_build_progress_event, drive_inference, enqueue_inference,
    handle_catalog_delete_requests, handle_ui_visibility_shortcut, processing_window_title,
    scene_bsn_export_for_world, scene_glb_export_for_world, should_run_headless_once,
    title_rattler_frame,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::app::{
    DepthDebugIntrinsics, InferenceDispatchGate, depth_debug_sample_stride,
    depth_debug_world_point, load_generated_glb_mesh_asset, scene_build_args_from_ui_settings,
    should_pause_render_during_inference, should_wait_before_inference_dispatch,
};
use bevy_synth_runtime::args::{
    AppArgs, BackendKind, DEFAULT_TRELLIS_PBR_TEXTURE_SIZE, DinoBackend, MeshMode, QualityPreset,
    RmbgBackend, RmbgModel, SynthesisModel, TrellisQuality, TripoSplatProfile, WeightPrecision,
};
use bevy_synth_runtime::cache::{CachedAssetAabb, CachedCameraState, CachedWorldItem, MeshCache};
#[cfg(not(target_arch = "wasm32"))]
use bevy_synth_runtime::io::write_glb;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceWorker, UiStatus, WorkerCommand, WorkerEvent,
};
use bevy_synth_runtime::{GaussianSplat, GaussianSplatCloud, SynthAsset, SynthMesh, TripoMesh};
use bevy_synth_ui::bevy_transform_gizmos::{GizmoTransformable, TransformGizmoOffset};

static TEST_CACHE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn test_args() -> AppArgs {
    AppArgs {
        image: None,
        prompt: None,
        text_embeds: None,
        text_embeds_key: "input.text_embeds".to_string(),
        weights_root: None,
        trellis_weights_root: None,
        triposplat_weights_root: None,
        trellis_image_large_root: None,
        trellis_quality: TrellisQuality::Low,
        trellis_pbr_enabled: true,
        trellis_pbr_texture_size: Some(DEFAULT_TRELLIS_PBR_TEXTURE_SIZE),
        trellis_target_faces: None,
        trellis_max_sparse_coords: None,
        scribble_weights_root: None,
        quality: QualityPreset::Full,
        triposplat_profile: TripoSplatProfile::Balanced,
        num_steps: 1,
        num_tokens: 4,
        guidance_scale: 1.0,
        triposplat_shift: 3.0,
        triposplat_num_gaussians: 262_144,
        triposplat_erode_radius: 1,
        seed: None,
        resolution: 16,
        chunk_size: 256,
        bounds: vec![-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
        mesh_mode: MeshMode::Flash,
        dense_octree_depth: 5,
        hierarchical_octree_depth: 6,
        band_threshold: 1.0,
        flash_octree_depth: 6,
        flash_min_resolution: 7,
        flash_mini_grid_num: 1,
        flash_num_chunks: 64,
        flash_mc_level: 0.0,
        target_faces: None,
        output: None,
        mesh: None,
        bg_weights_root: None,
        synthesis_models: vec![SynthesisModel::Triposg],
        available_synthesis_models: vec![SynthesisModel::Triposg, SynthesisModel::Triposplat],
        rmbg_model: RmbgModel::Rmbg14,
        backend: BackendKind::Cpu,
        rmbg_backend: RmbgBackend::Auto,
        dino_backend: DinoBackend::Auto,
        weights_precision: WeightPrecision::Auto,
        rmbg_weights_precision: WeightPrecision::Auto,
        pause_render_during_inference: true,
        ui_visible: true,
        read_only: false,
        max_batch_size: 1,
        mcp_scene_control_path: None,
        scene_bsn: None,
        scene_assets_json: None,
        scene_bsn_clear_existing: true,
    }
}

fn dummy_mesh() -> SynthMesh {
    SynthMesh::from(TripoMesh {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces: vec![[0, 1, 2]],
    })
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn depth_debug_sample_stride_caps_to_720p_budget() {
    let stride = depth_debug_sample_stride(3840, 2160, 1280 * 720);
    let sampled = 3840usize.div_ceil(stride) * 2160usize.div_ceil(stride);
    assert!(sampled <= 1280 * 720);
    assert_eq!(depth_debug_sample_stride(640, 360, 1280 * 720), 1);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn depth_debug_unprojects_center_pixel_along_scene_camera_forward() {
    let intrinsics = DepthDebugIntrinsics {
        fx: 100.0,
        fy: 100.0,
        cx: 1.5,
        cy: 1.5,
        width: 4,
        height: 4,
    };
    let transform = Transform::from_translation(Vec3::new(2.0, 1.0, -3.0))
        .looking_at(Vec3::new(2.0, 1.0, -4.0), Vec3::Y);
    let point = depth_debug_world_point(1, 1, 2.0, intrinsics, transform);
    assert!((point.x - 2.0).abs() < 1.0e-5);
    assert!((point.y - 1.0).abs() < 1.0e-5);
    assert!((point.z + 5.0).abs() < 1.0e-5);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_ui_stage_toggles_map_to_scene_build_args() {
    let mut settings = ScenePipelineUiSettings::default();
    assert_eq!(settings.quality_profile, SceneQualityProfileSetting::Fast);
    assert_eq!(settings.candidate_count, 1);
    assert_eq!(settings.feedback_iterations, 0);
    assert!(settings.pbr_enabled);
    assert_eq!(settings.target_faces, 80_000);
    assert!(!settings.feedback_enabled);
    let default_args = scene_build_args_from_ui_settings(
        PathBuf::from("scene.jpg"),
        Some(PathBuf::from("tmp/runs/test_scene_default")),
        1,
        vec![McpSynthesisModel::Trellis],
        &settings,
    );
    assert_eq!(
        default_args.composition_mode,
        SceneCompositionMode::CvGrounded
    );
    assert_eq!(default_args.pose_fit, ScenePoseFitMode::RenderedSilhouette);
    assert_eq!(default_args.canonical_pose, SceneCanonicalPoseMode::Off);
    assert_eq!(default_args.scale_policy, SceneScalePolicy::AssetPreserving);
    assert_eq!(default_args.depth_provider, SceneDepthProvider::DepthPro);
    assert_eq!(default_args.locator, SceneLocatorProvider::LocateAnything);
    assert_eq!(
        default_args.segmentation_provider,
        Some(SceneSegmentationProvider::Sam2)
    );
    assert!(!default_args.feedback);
    assert_eq!(default_args.feedback_iters, 0);
    assert_eq!(
        default_args.rotation_fit,
        burn_synth_mcp::SceneRotationFitMode::Off
    );
    assert_eq!(default_args.rotation_fit_max_gpt_rounds, 0);
    assert_eq!(
        default_args.object_pose_refinement,
        SceneObjectPoseRefinementMode::GatedGpt
    );
    assert_eq!(
        default_args.object_pose_refinement_set,
        SceneObjectPoseRefinementSet::TablesAndLargeSeating
    );

    settings.lift_assets = false;
    settings.locate_anything_enabled = false;
    settings.depth_enabled = false;
    settings.segmentation_enabled = true;
    settings.pose_fit_enabled = false;
    settings.feedback_enabled = true;
    settings.feedback_iterations = 4;
    settings.write_artifacts = false;
    settings.promote_to_catalog = false;

    let args = scene_build_args_from_ui_settings(
        PathBuf::from("scene.jpg"),
        Some(PathBuf::from("tmp/runs/test_scene")),
        0,
        vec![McpSynthesisModel::Trellis],
        &settings,
    );

    assert!(!args.lift_assets);
    assert!(!args.promote_to_catalog);
    assert_eq!(args.composition_mode, SceneCompositionMode::Heuristic);
    assert_eq!(args.canonical_pose, SceneCanonicalPoseMode::Off);
    assert_eq!(args.scale_policy, SceneScalePolicy::AssetPreserving);
    assert_eq!(args.depth_provider, SceneDepthProvider::None);
    assert_eq!(args.locator, SceneLocatorProvider::Manifest);
    assert!(args.locate_anything_backend.is_none());
    assert_eq!(
        args.segmentation_provider,
        Some(SceneSegmentationProvider::Sam2)
    );
    assert!(!args.write_artifacts);
    assert!(!args.save_pose_debug);
    assert!(
        !args.feedback,
        "feedback must stay disabled when asset lifting is disabled"
    );
    assert_eq!(args.feedback_iters, 0);
    assert_eq!(args.batch_size, Some(1));
    assert_eq!(args.quality_profile, Some(SceneQualityProfile::Draft));
    assert_eq!(args.trellis_pbr, Some(true));
    assert_eq!(args.target_faces, Some(80_000));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn generated_glb_path_loader_uses_runtime_mesh_parser() {
    let dir = isolated_cache_root();
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let path = dir.join("generated.glb");
    write_glb(&path, &dummy_mesh()).expect("write generated glb");

    let mut meshes = Assets::<BevyMesh>::default();
    let mut images = Assets::<Image>::default();
    let mut materials = Assets::<StandardMaterial>::default();
    let (mesh_handle, material_handle) =
        load_generated_glb_mesh_asset(&path, &mut meshes, &mut images, &mut materials)
            .expect("load generated glb");

    assert!(
        meshes.get(&mesh_handle).is_some(),
        "generated GLB should produce a Bevy mesh handle"
    );
    assert!(
        materials.get(&material_handle).is_some(),
        "generated GLB should produce a Bevy material handle"
    );
    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn startup_bsn_scene_writes_mcp_command_envelope() {
    let dir = isolated_cache_root();
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let bsn_path = dir.join("scene.bsn");
    let assets_path = dir.join("assets.json");
    let command_path = dir.join("scene_commands.json");
    let mesh_path = dir.join("chair.glb");
    write_glb(&mesh_path, &dummy_mesh()).expect("write generated glb");
    std::fs::write(
        &bsn_path,
        "synth_scene_v1 {\nasset chair_asset = \"generated:chair_asset\";\nspawn chair_left uses chair_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];\n}\n",
    )
    .expect("write bsn");
    std::fs::write(
        &assets_path,
        serde_json::to_vec_pretty(&serde_json::json!([
            {
                "asset_id": "chair_asset",
                "object_id": "chair_group",
                "label": "chair",
                "aliases": ["chair"],
                "path": mesh_path,
                "cache_key": null,
                "reusable": false,
                "source_image_path": null,
                "pipeline": "trellis",
                "provenance": null
            }
        ]))
        .unwrap(),
    )
    .expect("write assets");
    let mut args = test_args();
    args.scene_bsn = Some(bsn_path);
    args.scene_assets_json = Some(assets_path);
    args.mcp_scene_control_path = Some(command_path.clone());
    prepare_startup_bsn_scene(&mut args).expect("prepare startup bsn");
    assert_eq!(args.mcp_scene_control_path.as_ref(), Some(&command_path));
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&command_path).expect("read envelope")).unwrap();
    assert_eq!(
        envelope["session_id"],
        serde_json::json!("bevy_synth-startup-bsn")
    );
    assert_eq!(
        envelope["commands"][0]["type"],
        serde_json::json!("clear_scene")
    );
    assert_eq!(
        envelope["commands"][1]["type"],
        serde_json::json!("spawn_path")
    );
    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn startup_bsn_scene_accepts_self_contained_path_asset() {
    let dir = isolated_cache_root();
    std::fs::create_dir_all(&dir).expect("create temp dir");
    let bsn_path = dir.join("scene.bsn");
    let command_path = dir.join("scene_commands.json");
    let mesh_path = dir.join("chair.glb");
    write_glb(&mesh_path, &dummy_mesh()).expect("write generated glb");
    std::fs::write(
        &bsn_path,
        format!(
            "synth_scene_v1 {{\nasset chair_asset = \"path:{}\";\nspawn chair_left uses chair_asset translation [0.0,0.0,0.0] rotation_y 0.0 scale [1.0,1.0,1.0];\n}}\n",
            mesh_path.display()
        ),
    )
    .expect("write bsn");
    let mut args = test_args();
    args.scene_bsn = Some(bsn_path);
    args.scene_assets_json = None;
    args.mcp_scene_control_path = Some(command_path.clone());
    prepare_startup_bsn_scene(&mut args).expect("prepare startup bsn");
    let envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&command_path).expect("read envelope")).unwrap();
    assert_eq!(
        envelope["commands"][1]["type"],
        serde_json::json!("spawn_path")
    );
    assert_eq!(
        envelope["commands"][1]["path"],
        serde_json::json!(mesh_path.to_string_lossy())
    );
    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_bsn_export_serializes_only_placed_cached_assets() {
    let dir = isolated_cache_root();
    let mut cache = MeshCache::load_from_root(dir.clone()).expect("create isolated cache");
    let chair = cache
        .upsert_mesh_for_image(&PathBuf::from("chair.png"), &dummy_mesh())
        .expect("cache chair mesh");
    let unused = cache
        .upsert_mesh_for_image(&PathBuf::from("unused.png"), &dummy_mesh())
        .expect("cache unused mesh");
    let world_items = vec![CachedWorldItem {
        cache_key: chair.cache_key.clone(),
        translation: [1.0, 0.0, 2.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.25, 1.0, 1.25],
    }];
    let camera = CachedCameraState {
        translation: [0.0, 3.0, 5.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        focus: [0.0, 0.0, 0.0],
        yaw: 0.25,
        pitch: -0.5,
        radius: 5.8,
        vertical_fov_degrees: Some(72.0),
    };

    let (bsn, assets_json) =
        scene_bsn_export_for_world(&cache, &world_items, Some(camera)).expect("scene bsn");
    assert!(bsn.contains("synth_scene_v1"));
    assert!(bsn.contains("spawn item_001_"));
    assert!(bsn.contains("camera translation"));
    assert!(bsn.contains(&chair.cache_key));
    assert!(!bsn.contains(&unused.cache_key));

    let assets: Vec<burn_synth_scene::SceneAssetBinding> =
        serde_json::from_slice(&assets_json).expect("asset sidecar json");
    assert_eq!(assets.len(), 1);
    assert_eq!(
        assets[0].cache_key.as_deref(),
        Some(chair.cache_key.as_str())
    );
    assert_eq!(assets[0].source_image_path.as_deref(), Some("chair.png"));

    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_glb_export_rejects_splats_and_exports_mesh_instances() {
    let dir = isolated_cache_root();
    let mut cache = MeshCache::load_from_root(dir.clone()).expect("create isolated cache");
    let chair = cache
        .upsert_mesh_for_image(&PathBuf::from("chair.png"), &dummy_mesh())
        .expect("cache chair mesh");
    let splat = cache
        .upsert_gaussian_splat_for_image(
            &PathBuf::from("cloud.png"),
            &GaussianSplatCloud::canonical_debug_cloud(),
        )
        .expect("cache splat");

    let mesh_items = vec![CachedWorldItem {
        cache_key: chair.cache_key.clone(),
        translation: [1.0, 0.0, 2.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }];
    let glb = scene_glb_export_for_world(&cache, &mesh_items).expect("mesh scene glb");
    assert!(
        glb.starts_with(&[0x67, 0x6C, 0x54, 0x46]),
        "mesh scene export should be a binary GLB"
    );

    let splat_items = vec![CachedWorldItem {
        cache_key: splat.cache_key.clone(),
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    }];
    let err = scene_glb_export_for_world(&cache, &splat_items).unwrap_err();
    assert!(err.contains("mesh scenes only"));

    std::fs::remove_dir_all(dir).expect("remove temp dir");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn projected_feedback_uses_explicit_bounds_for_fresh_path_assets() {
    let item = CachedWorldItem {
        cache_key: "generated-chair".to_string(),
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };
    let aabb = CachedAssetAabb {
        min: [-0.5, 0.0, -0.5],
        max: [0.5, 1.0, 0.5],
    };
    let metadata_by_key = std::collections::HashMap::new();
    let mut explicit_bounds = std::collections::HashMap::new();
    explicit_bounds.insert(item.cache_key.clone(), aabb);

    assert_eq!(
        crate::app::world_item_local_aabb(&metadata_by_key, &explicit_bounds, &item),
        Some(aabb)
    );
}

fn isolated_cache_root() -> PathBuf {
    let nonce = TEST_CACHE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "bevy_synth_test_cache_{}_{}_{}",
        std::process::id(),
        now,
        nonce
    ))
}

fn build_test_app(worker: InferenceWorker, queue: InferenceQueue, status: UiStatus) -> App {
    let mut app = App::new();
    app.insert_resource(test_args());
    app.insert_resource(queue);
    app.insert_resource(worker);
    app.insert_resource(status);
    #[cfg(not(target_arch = "wasm32"))]
    app.insert_resource(InferenceDispatchGate::ready_for_dispatch());
    app.insert_resource(CatalogState::default());
    app.insert_resource(ExitState::default());
    app.insert_resource(SceneInteractionLock::default());
    app.insert_resource(SceneReadOnlyMode::default());
    app.insert_resource(UiVisibilityState::new(true));
    let cache = MeshCache::load_from_root(isolated_cache_root()).expect("create isolated cache");
    app.insert_resource(MeshCacheResource { cache });
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(Assets::<StandardMaterial>::default());
    app.insert_resource(Assets::<PlanarGaussian3d>::default());
    app.add_systems(
        Update,
        (drive_inference, crate::app::sync_gaussian_splat_pick_bounds).chain(),
    );
    app
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_progress_event_updates_processing_status() {
    let mut processing = SceneProcessingState::default();
    processing.begin("scene.jpg");
    let mut status = UiStatus {
        message: String::new(),
        processing: false,
        worker_message: None,
    };
    apply_scene_build_progress_event(
        SceneBuildProgressEvent {
            run_id: "run-1".to_string(),
            sequence: 1,
            stage: "depth_pro_grounding_evidence".to_string(),
            phase: SceneBuildProgressPhase::Waiting,
            execution: SceneBuildExecutionKind::Gpu,
            message: "running DepthPro".to_string(),
            elapsed_ms: 1200,
            item_index: None,
            item_count: Some(1),
            artifact_path: Some("tmp/runs/run-1/depth_pro/depth_evidence.json".to_string()),
            detail: serde_json::json!({
                "has_depth": true,
                "token_usage": {
                    "total": {
                        "requests": 2,
                        "reported_requests": 2,
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "total_tokens": 150
                    },
                    "by_stage": [
                        { "stage": "plan_objects", "total_tokens": 90 },
                        { "stage": "generate_object_images", "total_tokens": 60 }
                    ]
                }
            }),
        },
        &mut processing,
        &mut status,
    );

    let worker_message = status.worker_message.expect("worker message");
    assert!(worker_message.contains("waiting"));
    assert!(worker_message.contains("depth_pro_grounding_evidence"));
    assert!(processing.is_visible());
    assert!(
        processing
            .token_usage_summary()
            .is_some_and(|summary| summary.contains("tokens total=150"))
    );
}

#[test]
fn inference_queue_advances_and_tracks_completed() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let worker = InferenceWorker {
        sender: cmd_tx,
        receiver: Mutex::new(event_rx),
    };

    let mut queue = InferenceQueue::default();
    let args = test_args();
    enqueue_inference(PathBuf::from("first.png"), &args, &mut queue);
    enqueue_inference(PathBuf::from("second.png"), &args, &mut queue);
    let status = UiStatus {
        message: String::new(),
        processing: false,
        worker_message: None,
    };

    let mut app = build_test_app(worker, queue, status);
    app.update();

    let queue = app.world().resource::<InferenceQueue>();
    assert!(queue.active.is_some());
    assert_eq!(queue.pending.len(), 1);

    let command = cmd_rx.try_recv().expect("expected infer command");
    let WorkerCommand::Infer(batch) = command else {
        panic!("expected infer command");
    };
    assert_eq!(batch.len(), 1);
    let first_request = batch[0].clone();

    event_tx
        .send(WorkerEvent {
            requests: vec![first_request.clone()],
            results: vec![Ok(None)],
            elapsed: Duration::from_millis(1),
            status_message: None,
        })
        .expect("send worker event");

    app.update();

    let queue = app.world().resource::<InferenceQueue>();
    assert_eq!(queue.completed, 1);
    assert!(queue.active.is_some());
    assert!(queue.pending.is_empty());
}

#[test]
fn inference_queue_dispatches_batches_up_to_configured_limit() {
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let (_event_tx, event_rx) = mpsc::channel();
    let worker = InferenceWorker {
        sender: cmd_tx,
        receiver: Mutex::new(event_rx),
    };

    let mut queue = InferenceQueue::default();
    let args = test_args();
    enqueue_inference(PathBuf::from("first.png"), &args, &mut queue);
    enqueue_inference(PathBuf::from("second.png"), &args, &mut queue);
    enqueue_inference(PathBuf::from("third.png"), &args, &mut queue);
    let status = UiStatus {
        message: String::new(),
        processing: false,
        worker_message: None,
    };

    let mut app = build_test_app(worker, queue, status);
    app.world_mut().resource_mut::<AppArgs>().max_batch_size = 2;
    app.update();

    let command = cmd_rx.try_recv().expect("expected batched infer command");
    let WorkerCommand::Infer(batch) = command else {
        panic!("expected infer command");
    };
    assert_eq!(
        batch
            .iter()
            .map(|request| request.image_path.clone())
            .collect::<Vec<_>>(),
        vec![PathBuf::from("first.png"), PathBuf::from("second.png")]
    );

    let queue = app.world().resource::<InferenceQueue>();
    assert_eq!(queue.active.as_ref().map(Vec::len), Some(2));
    assert_eq!(queue.pending.len(), 1);
}

#[test]
fn enqueue_inference_snapshots_triposplat_settings() {
    let mut queue = InferenceQueue::default();
    let mut args = test_args();
    args.synthesis_models = vec![SynthesisModel::Triposplat];
    args.num_steps = 5;
    args.num_tokens = 768;
    args.guidance_scale = 3.0;
    args.target_faces = Some(8_000);
    args.triposplat_num_gaussians = 32_768;

    let request = enqueue_inference(PathBuf::from("splat.png"), &args, &mut queue);
    args.num_steps = 50;
    args.num_tokens = 2048;
    args.guidance_scale = 4.5;
    args.target_faces = Some(20_000);
    args.triposplat_num_gaussians = 262_144;

    assert_eq!(request.settings.num_steps, 5);
    assert_eq!(request.settings.num_tokens, 768);
    assert_eq!(request.settings.guidance_scale, 3.0);
    assert_eq!(request.settings.target_faces, Some(8_000));
    assert_eq!(request.settings.triposplat_num_gaussians, 32_768);
}

#[test]
fn processing_window_title_uses_constant_width_rattler() {
    let titles = (0..8)
        .map(|phase| processing_window_title("image.png", 2, phase))
        .collect::<Vec<_>>();
    let first_len = titles[0].len();

    assert!(titles.iter().all(|title| title.len() == first_len));
    assert!(titles.iter().all(|title| title.starts_with("bevy_synth [")));
    assert!(titles.iter().all(|title| title.ends_with("(queued: 2)")));
    assert!((0..8).all(|phase| title_rattler_frame(phase).len() == 3));
}

#[test]
fn inference_result_spawns_mesh_entity() {
    let (cmd_tx, _cmd_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let worker = InferenceWorker {
        sender: cmd_tx,
        receiver: Mutex::new(event_rx),
    };

    let mut queue = InferenceQueue::default();
    let args = test_args();
    enqueue_inference(PathBuf::from("mesh.png"), &args, &mut queue);
    queue.active = Some(vec![queue.pending.pop_front().expect("pending request")]);
    let status = UiStatus {
        message: String::new(),
        processing: true,
        worker_message: None,
    };

    let request = queue
        .active
        .as_ref()
        .and_then(|batch| batch.first())
        .cloned()
        .expect("active request");
    let mut app = build_test_app(worker, queue, status);

    event_tx
        .send(WorkerEvent {
            requests: vec![request],
            results: vec![Ok(Some(SynthAsset::Mesh(dummy_mesh())))],
            elapsed: Duration::from_millis(1),
            status_message: None,
        })
        .expect("send worker event");

    app.update();

    let world = app.world_mut();
    let count = world.query::<&GizmoTransformable>().iter(world).count();
    assert_eq!(count, 1);
}

#[test]
fn inference_result_with_splats_writes_output_and_spawns_gaussian_cloud() {
    let (cmd_tx, _cmd_rx) = mpsc::channel();
    let (event_tx, event_rx) = mpsc::channel();
    let worker = InferenceWorker {
        sender: cmd_tx,
        receiver: Mutex::new(event_rx),
    };

    let mut queue = InferenceQueue::default();
    let args = test_args();
    enqueue_inference(PathBuf::from("splat.png"), &args, &mut queue);
    queue.active = Some(vec![queue.pending.pop_front().expect("pending request")]);
    let status = UiStatus {
        message: String::new(),
        processing: true,
        worker_message: None,
    };

    let mut request = queue
        .active
        .as_ref()
        .and_then(|batch| batch.first())
        .cloned()
        .expect("active request");
    let request_id = request.id;
    let output_path = isolated_cache_root().join("debug.splat");
    request.output_path = Some(output_path.clone());
    let splats = GaussianSplatCloud::canonical_debug_cloud();
    let expected_bytes = splats.stats().splat_bytes;
    let mut app = build_test_app(worker, queue, status);
    app.world_mut()
        .resource_mut::<CatalogState>()
        .add_pending(&request);

    event_tx
        .send(WorkerEvent {
            requests: vec![request],
            results: vec![Ok(Some(SynthAsset::GaussianSplat(splats)))],
            elapsed: Duration::from_millis(1),
            status_message: None,
        })
        .expect("send worker event");

    app.update();

    let world = app.world_mut();
    let count = world.query::<&GizmoTransformable>().iter(world).count();
    assert_eq!(count, 1);
    let cloud_entities = world.query::<&PlanarGaussian3dHandle>().iter(world).count();
    assert_eq!(cloud_entities, 1);
    let mut pick_bounds =
        world.query::<(&crate::app::GaussianSplatPickBounds, &TransformGizmoOffset)>();
    let (bounds, offset) = pick_bounds
        .single(world)
        .expect("Gaussian cloud should expose pick bounds for selection");
    assert_eq!(offset.0, bounds.center);
    let mut settings = world.query::<&CloudSettings>();
    let settings = settings.single(world).expect("one Gaussian cloud settings");
    assert_eq!(settings.sort_mode, SortMode::Std);
    assert_eq!(settings.color_space, GaussianColorSpace::SrgbRec709Display);
    let mesh_entities = world.query::<&Mesh3d>().iter(world).count();
    assert_eq!(mesh_entities, 0, "TripoSplat should not spawn a mesh proxy");

    {
        let catalog = world.resource::<CatalogState>();
        let entry = catalog.entry(request_id).expect("catalog entry");
        assert!(matches!(entry.status, CatalogStatus::Ready));
        assert!(
            entry.cache_key.is_some(),
            "splat renderer entity should be cache-backed"
        );
        assert!(entry.mesh.is_none());
        assert!(entry.material.is_none());
        assert!(
            entry.gaussian.is_some(),
            "catalog entry should carry the splat cloud for preview and respawn"
        );
    }
    let gaussian_clouds = world.resource::<Assets<PlanarGaussian3d>>();
    assert_eq!(gaussian_clouds.iter().count(), 1);
    assert_eq!(
        std::fs::metadata(&output_path)
            .expect("splat output metadata")
            .len(),
        expected_bytes as u64
    );
}

#[test]
fn gaussian_splat_cloud_conversion_preserves_full_cloud_count() {
    let splat = GaussianSplat {
        position: [0.0, 0.0, 0.0],
        features_dc: [0.0, 0.0, 0.0],
        opacity: 0.5,
        scale: [0.01, 0.01, 0.01],
        rotation: [1.0, 0.0, 0.0, 0.0],
    };
    let splats = GaussianSplatCloud::new(vec![splat; 8_193]);

    let cloud = crate::app::gaussian_splat_cloud_to_planar_gaussian_3d(&splats)
        .expect("build Gaussian cloud");

    assert_eq!(cloud.position_visibility.len(), 8_193);
}

#[test]
fn gaussian_splat_cloud_conversion_uses_bevy_display_orientation() {
    let splats = GaussianSplatCloud::new(vec![GaussianSplat {
        position: [1.0, 2.0, 3.0],
        features_dc: [0.1, 0.2, 0.3],
        opacity: 0.5,
        scale: [0.01, 0.02, 0.03],
        rotation: [1.0, 0.0, 0.0, 0.0],
    }]);

    let cloud =
        crate::app::gaussian_splat_cloud_to_planar_gaussian_3d(&splats).expect("build cloud");

    assert_eq!(cloud.position_visibility[0].position, [2.0, 3.0, 1.0]);
    assert_eq!(cloud.position_visibility[0].visibility, 1.0);
    assert_eq!(cloud.scale_opacity[0].scale, [0.01, 0.02, 0.03]);
    assert_eq!(cloud.scale_opacity[0].opacity, 0.5);
    assert_eq!(
        cloud.spherical_harmonic[0].coefficients[0..3],
        [0.1, 0.2, 0.3]
    );
}

#[test]
fn triposplat_cloud_settings_use_display_rgb_color_space() {
    let settings = crate::app::triposplat_cloud_settings();
    assert_eq!(settings.sort_mode, SortMode::Std);
    assert_eq!(settings.color_space, GaussianColorSpace::SrgbRec709Display);
}

#[test]
fn gaussian_splat_pick_bounds_cover_cloud_extent() {
    let cloud = PlanarGaussian3d::from(vec![
        bevy_gaussian_splatting::Gaussian3d {
            position_visibility: [-1.0, 0.0, 0.0, 1.0].into(),
            spherical_harmonic: Default::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.1, 0.2, 0.3, 0.8].into(),
        },
        bevy_gaussian_splatting::Gaussian3d {
            position_visibility: [1.0, 2.0, 3.0, 1.0].into(),
            spherical_harmonic: Default::default(),
            rotation: [1.0, 0.0, 0.0, 0.0].into(),
            scale_opacity: [0.2, 0.1, 0.1, 0.8].into(),
        },
    ]);

    let bounds = crate::app::gaussian_splat_pick_bounds(&cloud).expect("bounds");
    let (world_min, world_max) = crate::app::world_aabb(
        bounds.center,
        bounds.half_extents,
        &GlobalTransform::IDENTITY,
    );

    assert!(world_min.x < -1.0);
    assert!(world_max.x > 1.0);
    assert!(world_max.y > 2.0);
    assert!(world_max.z > 3.0);
    assert!(
        crate::app::ray_aabb_intersection(
            Vec3::new(0.0, 1.0, 6.0),
            Vec3::new(0.0, 0.0, -1.0),
            world_min,
            world_max,
        )
        .is_some(),
        "Gaussian cloud bounds should be usable as a click target"
    );
}

#[test]
fn viewer_debug_entity_visibility_modes_match_settings() {
    assert!(!crate::app::viewer_debug_entity_visible(
        ViewerAabbOverlayMode::Off,
        true
    ));
    assert!(crate::app::viewer_debug_entity_visible(
        ViewerAabbOverlayMode::Selected,
        true
    ));
    assert!(!crate::app::viewer_debug_entity_visible(
        ViewerAabbOverlayMode::Selected,
        false
    ));
    assert!(crate::app::viewer_debug_entity_visible(
        ViewerAabbOverlayMode::All,
        false
    ));
}

#[test]
fn viewer_ground_contact_state_classifies_gap_direction() {
    assert_eq!(
        crate::app::viewer_ground_contact_state(0.01, 0.0, 0.02),
        crate::app::ViewerGroundContactState::Grounded
    );
    assert_eq!(
        crate::app::viewer_ground_contact_state(0.05, 0.0, 0.02),
        crate::app::ViewerGroundContactState::Floating
    );
    assert_eq!(
        crate::app::viewer_ground_contact_state(-0.05, 0.0, 0.02),
        crate::app::ViewerGroundContactState::BelowGround
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn scene_camera_frustum_debug_gizmo_is_bounded() {
    assert_eq!(crate::app::scene_camera_frustum_length(0.001), 0.05);
    assert_eq!(crate::app::scene_camera_frustum_length(100.0), 3.0);

    let default_cross = crate::app::scene_camera_frustum_cross_size(0.75);
    assert!((0.02..=0.08).contains(&default_cross));
    assert!(
        default_cross <= 0.03,
        "default scene camera cross should stay visually small, got {default_cross}"
    );
}

#[test]
fn ui_plugin_update_has_no_query_conflicts() {
    let mut app = App::new();
    app.insert_resource(InferenceQueue::default());
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(Assets::<PlanarGaussian3d>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(Time::<()>::default());
    app.add_plugins(BurnSynthUiPlugin);

    app.update();
}

#[test]
fn f1_toggles_ui_visibility_state() {
    let mut app = App::new();
    app.insert_resource(UiVisibilityState::new(true));
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.add_systems(Update, handle_ui_visibility_shortcut);

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::F1);
    app.update();

    assert!(!app.world().resource::<UiVisibilityState>().visible);

    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .reset_all();
    app.world_mut()
        .resource_mut::<ButtonInput<KeyCode>>()
        .press(KeyCode::F1);
    app.update();

    assert!(app.world().resource::<UiVisibilityState>().visible);
}

fn write_catalog_delete_request_once(
    mut wrote: Local<bool>,
    mut requests: MessageWriter<CatalogDeleteRequest>,
) {
    if *wrote {
        return;
    }
    *wrote = true;
    requests.write(CatalogDeleteRequest {
        cache_key: Some("cache-key".to_string()),
    });
}

#[test]
fn catalog_delete_request_removes_cache_backed_instances_in_same_update() {
    let mut app = App::new();
    app.add_message::<CatalogDeleteRequest>();
    app.insert_resource(MeshCacheResource {
        cache: MeshCache::load_from_root(isolated_cache_root()).expect("create isolated cache"),
    });
    app.insert_resource(SceneInteractionLock::default());
    app.insert_resource(SceneReadOnlyMode::default());
    let entity = app
        .world_mut()
        .spawn(CachedMeshInstance {
            cache_key: "cache-key".to_string(),
            local_aabb: None,
        })
        .id();
    app.add_systems(
        Update,
        (
            write_catalog_delete_request_once.in_set(BurnSynthUiSystemSet::CatalogRequests),
            handle_catalog_delete_requests.after(BurnSynthUiSystemSet::CatalogRequests),
        ),
    );

    app.update();

    assert!(
        !app.world().entities().contains(entity),
        "cache-backed spawned instance should be despawned on the same update as the catalog delete request"
    );
}

#[test]
fn catalog_delete_request_is_ignored_while_scene_interaction_locked() {
    let mut app = App::new();
    app.add_message::<CatalogDeleteRequest>();
    app.insert_resource(MeshCacheResource {
        cache: MeshCache::load_from_root(isolated_cache_root()).expect("create isolated cache"),
    });
    let mut interaction_lock = SceneInteractionLock::default();
    interaction_lock.set(true, Some("feedback iteration".to_string()));
    app.insert_resource(interaction_lock);
    app.insert_resource(SceneReadOnlyMode::default());
    let entity = app
        .world_mut()
        .spawn(CachedMeshInstance {
            cache_key: "cache-key".to_string(),
            local_aabb: None,
        })
        .id();
    app.add_systems(
        Update,
        (
            write_catalog_delete_request_once.in_set(BurnSynthUiSystemSet::CatalogRequests),
            handle_catalog_delete_requests.after(BurnSynthUiSystemSet::CatalogRequests),
        ),
    );

    app.update();

    assert!(
        app.world().entities().contains(entity),
        "read-only scene interaction lock should prevent catalog deletes from changing the scene"
    );
}

#[test]
fn catalog_delete_request_is_ignored_in_read_only_mode() {
    let mut app = App::new();
    app.add_message::<CatalogDeleteRequest>();
    app.insert_resource(MeshCacheResource {
        cache: MeshCache::load_from_root(isolated_cache_root()).expect("create isolated cache"),
    });
    app.insert_resource(SceneInteractionLock::default());
    app.insert_resource(SceneReadOnlyMode { enabled: true });
    let entity = app
        .world_mut()
        .spawn(CachedMeshInstance {
            cache_key: "cache-key".to_string(),
            local_aabb: None,
        })
        .id();
    app.add_systems(
        Update,
        (
            write_catalog_delete_request_once.in_set(BurnSynthUiSystemSet::CatalogRequests),
            handle_catalog_delete_requests.after(BurnSynthUiSystemSet::CatalogRequests),
        ),
    );

    app.update();

    assert!(
        app.world().entities().contains(entity),
        "read-only mode should prevent catalog deletes from changing the scene"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn headless_once_requires_image_and_output_without_mesh() {
    let mut args = test_args();
    assert!(!should_run_headless_once(&args));

    args.image = Some(PathBuf::from("docs/input_chair.jpg"));
    assert!(!should_run_headless_once(&args));

    args.output = Some(PathBuf::from("docs/output.glb"));
    assert!(should_run_headless_once(&args));

    args.mesh = Some(PathBuf::from("docs/output.glb"));
    assert!(!should_run_headless_once(&args));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn render_pause_toggle_follows_queue_state() {
    let mut args = test_args();
    let mut queue = InferenceQueue::default();

    assert!(!should_pause_render_during_inference(&args, &queue, false));
    enqueue_inference(PathBuf::from("chair.png"), &args, &mut queue);
    assert!(!should_pause_render_during_inference(&args, &queue, false));
    queue.active = Some(vec![queue.pending.pop_front().expect("pending request")]);
    assert!(should_pause_render_during_inference(&args, &queue, false));
    assert!(!should_pause_render_during_inference(&args, &queue, true));

    args.pause_render_during_inference = false;
    assert!(!should_pause_render_during_inference(&args, &queue, false));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn inference_dispatch_gate_waits_for_visible_startup_frames() {
    let args = test_args();
    let mut queue = InferenceQueue::default();
    enqueue_inference(PathBuf::from("chair.png"), &args, &mut queue);
    let mut gate = InferenceDispatchGate::default();

    assert!(should_wait_before_inference_dispatch(&mut gate, &queue));
    assert!(should_wait_before_inference_dispatch(&mut gate, &queue));
    assert!(should_wait_before_inference_dispatch(&mut gate, &queue));
    assert!(!should_wait_before_inference_dispatch(&mut gate, &queue));

    queue.active = Some(vec![queue.pending.pop_front().expect("pending request")]);
    assert!(!should_wait_before_inference_dispatch(&mut gate, &queue));
    queue.active = None;
    queue.pending.clear();
    assert!(!should_wait_before_inference_dispatch(&mut gate, &queue));

    enqueue_inference(PathBuf::from("next.png"), &args, &mut queue);
    assert!(should_wait_before_inference_dispatch(&mut gate, &queue));
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn linux_full_flash_workload_uses_shared_wgpu_device_when_versions_align() {
    let mut args = test_args();
    args.backend = BackendKind::Wgpu;
    args.mesh_mode = MeshMode::Flash;
    args.flash_octree_depth = 9;
    args.flash_min_resolution = 63;
    assert!(
        should_share_wgpu_inference_device_for_platform(&args, true),
        "Bevy and Burn both use wgpu 29, so the WGPU inference device should be shared"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn current_wgpu_stack_uses_shared_device_for_lighter_workloads() {
    let mut args = test_args();
    args.backend = BackendKind::Wgpu;
    args.mesh_mode = MeshMode::Flash;
    args.flash_octree_depth = 8;
    args.flash_min_resolution = 31;

    assert!(
        should_share_wgpu_inference_device_for_platform(&args, true),
        "Bevy and Burn both use wgpu 29, so native WGPU inference should share the render device"
    );
    assert!(
        should_share_wgpu_inference_device_for_platform(&args, false),
        "Bevy and Burn both use wgpu 29, so native WGPU inference should share the render device"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn trellis_capable_sessions_use_isolated_wgpu_inference_device() {
    let mut args = test_args();
    args.backend = BackendKind::Wgpu;
    args.available_synthesis_models = vec![
        SynthesisModel::Triposg,
        SynthesisModel::Trellis,
        SynthesisModel::Triposplat,
    ];

    assert!(
        !should_share_wgpu_inference_device_for_platform(&args, true),
        "TRELLIS scene builds use an isolated Burn WGPU runtime to avoid sharing the Bevy render device with heavy model loading"
    );
}

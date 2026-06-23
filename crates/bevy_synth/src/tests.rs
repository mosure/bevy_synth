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
};

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use crate::app::should_share_wgpu_inference_device_for_platform;
use crate::app::{
    CachedMeshInstance, MeshCacheResource, drive_inference, enqueue_inference,
    handle_catalog_delete_requests, processing_window_title, should_run_headless_once,
    title_rattler_frame,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::app::{
    InferenceDispatchGate, load_generated_glb_mesh_asset, should_pause_render_during_inference,
    should_wait_before_inference_dispatch,
};
use bevy_synth_runtime::args::{
    AppArgs, BackendKind, DEFAULT_TRELLIS_PBR_TEXTURE_SIZE, DinoBackend, MeshMode, QualityPreset,
    RmbgBackend, RmbgModel, SynthesisModel, TrellisQuality, TripoSplatProfile, WeightPrecision,
};
use bevy_synth_runtime::cache::MeshCache;
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
        trellis_python_bin: None,
        trellis_bridge_script: None,
        trellis_quality: TrellisQuality::Low,
        trellis_pbr_enabled: false,
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
        max_batch_size: 1,
        mcp_scene_control_path: None,
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
    let entity = app
        .world_mut()
        .spawn(CachedMeshInstance {
            cache_key: "cache-key".to_string(),
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

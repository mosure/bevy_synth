use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use bevy::prelude::*;
use bevy_mesh::Mesh as BevyMesh;
use bevy_synth_ui::{BurnSynthUiPlugin, CatalogState};

#[cfg(not(target_arch = "wasm32"))]
use crate::app::should_pause_render_during_inference;
#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
use crate::app::should_share_wgpu_inference_device_for_platform;
use crate::app::{MeshCacheResource, drive_inference, enqueue_inference, should_run_headless_once};
use bevy_synth_runtime::args::{
    AppArgs, BackendKind, DinoBackend, MeshMode, QualityPreset, RmbgBackend, RmbgModel,
    SynthesisModel, TrellisQuality, WeightPrecision,
};
use bevy_synth_runtime::cache::MeshCache;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceWorker, UiStatus, WorkerCommand, WorkerEvent,
};
use bevy_synth_runtime::{SynthMesh, TripoMesh};
use bevy_synth_ui::bevy_transform_gizmos::GizmoTransformable;

static TEST_CACHE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn test_args() -> AppArgs {
    AppArgs {
        image: None,
        prompt: None,
        text_embeds: None,
        text_embeds_key: "input.text_embeds".to_string(),
        weights_root: None,
        trellis_weights_root: None,
        trellis_image_large_root: None,
        trellis_python_bin: None,
        trellis_bridge_script: None,
        trellis_quality: TrellisQuality::Medium,
        scribble_weights_root: None,
        quality: QualityPreset::Full,
        num_steps: 1,
        num_tokens: 4,
        guidance_scale: 1.0,
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
    app.insert_resource(CatalogState::default());
    app.insert_resource(ExitState::default());
    let cache = MeshCache::load_from_root(isolated_cache_root()).expect("create isolated cache");
    app.insert_resource(MeshCacheResource { cache });
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(Assets::<StandardMaterial>::default());
    app.add_systems(Update, drive_inference);
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
            results: vec![Ok(Some(dummy_mesh()))],
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
fn ui_plugin_update_has_no_query_conflicts() {
    let mut app = App::new();
    app.insert_resource(InferenceQueue::default());
    app.insert_resource(Assets::<Image>::default());
    app.insert_resource(Assets::<BevyMesh>::default());
    app.insert_resource(ButtonInput::<MouseButton>::default());
    app.insert_resource(ButtonInput::<KeyCode>::default());
    app.insert_resource(Time::<()>::default());
    app.add_plugins(BurnSynthUiPlugin);

    app.update();
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
    assert!(should_pause_render_during_inference(&args, &queue, false));
    assert!(!should_pause_render_during_inference(&args, &queue, true));

    args.pause_render_during_inference = false;
    assert!(!should_pause_render_during_inference(&args, &queue, false));
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn linux_full_flash_workload_uses_isolated_wgpu_device() {
    let mut args = test_args();
    args.backend = BackendKind::Wgpu;
    args.mesh_mode = MeshMode::Flash;
    args.flash_octree_depth = 9;
    args.flash_min_resolution = 63;
    assert!(
        !should_share_wgpu_inference_device_for_platform(&args, true),
        "Linux full+flash should isolate Burn WGPU device from Bevy render device"
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "wgpu"))]
#[test]
fn non_linux_or_lighter_flash_workloads_keep_shared_wgpu_device() {
    let mut args = test_args();
    args.backend = BackendKind::Wgpu;
    args.mesh_mode = MeshMode::Flash;
    args.flash_octree_depth = 8;
    args.flash_min_resolution = 31;

    assert!(
        should_share_wgpu_inference_device_for_platform(&args, true),
        "Linux lower flash workloads should continue to share the Bevy render device"
    );
    assert!(
        should_share_wgpu_inference_device_for_platform(&args, false),
        "Non-Linux platforms should continue to share the Bevy render device"
    );
}

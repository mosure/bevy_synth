use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;

use bevy::prelude::*;
use bevy_mesh::Mesh as BevyMesh;
use bevy_synth_ui::{BurnSynthUiPlugin, CatalogState};

use crate::app::{MeshCacheResource, drive_inference, enqueue_inference};
use bevy_synth_runtime::TripoMesh;
use bevy_synth_runtime::args::{
    AppArgs, BackendKind, DinoBackend, MeshMode, RmbgBackend, RmbgModel, SynthesisModel,
};
use bevy_synth_runtime::cache::MeshCache;
use bevy_synth_runtime::state::{
    ExitState, InferenceQueue, InferenceWorker, UiStatus, WorkerCommand, WorkerEvent,
};
use bevy_transform_gizmos::GizmoTransformable;

fn test_args() -> AppArgs {
    AppArgs {
        image: None,
        prompt: None,
        text_embeds: None,
        text_embeds_key: "input.text_embeds".to_string(),
        weights_root: None,
        scribble_weights_root: None,
        num_steps: 1,
        num_tokens: 4,
        guidance_scale: 1.0,
        seed: None,
        match_python: false,
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
        max_batch_size: 1,
    }
}

fn dummy_mesh() -> TripoMesh {
    TripoMesh {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        faces: vec![[0, 1, 2]],
    }
}

fn build_test_app(worker: InferenceWorker, queue: InferenceQueue, status: UiStatus) -> App {
    let mut app = App::new();
    app.insert_resource(test_args());
    app.insert_resource(queue);
    app.insert_resource(worker);
    app.insert_resource(status);
    app.insert_resource(CatalogState::default());
    app.insert_resource(ExitState::default());
    app.insert_resource(MeshCacheResource {
        cache: MeshCache::empty_default(),
    });
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
    app.insert_resource(Time::<()>::default());
    app.add_plugins(BurnSynthUiPlugin);

    app.update();
}

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;

use bevy::prelude::*;
use bevy_mesh::Mesh as BevyMesh;

use burn_3d_synth_tripo::pipeline::mesh::Mesh as TripoMesh;

use crate::app::{drive_inference, enqueue_inference};
use crate::args::{AppArgs, BackendKind, DinoBackend, MeshMode, RmbgBackend};
use crate::state::{
    DraggableMesh, ExitState, InferenceQueue, InferenceWorker, UiStatus, WorkerCommand,
    WorkerEvent,
};

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
        backend: BackendKind::Cpu,
        rmbg_backend: RmbgBackend::Auto,
        dino_backend: DinoBackend::Auto,
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
    app.insert_resource(ExitState::default());
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
    };

    let mut app = build_test_app(worker, queue, status);
    app.update();

    let queue = app.world().resource::<InferenceQueue>();
    assert!(queue.active.is_some());
    assert_eq!(queue.pending.len(), 1);

    let command = cmd_rx.try_recv().expect("expected infer command");
    let WorkerCommand::Infer(first_request) = command else {
        panic!("expected infer command");
    };

    event_tx
        .send(WorkerEvent {
            request: first_request.clone(),
            result: Ok(None),
            elapsed: Duration::from_millis(1),
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
    queue.active = queue.pending.pop_front();
    let status = UiStatus {
        message: String::new(),
        processing: true,
    };

    let request = queue.active.clone().expect("active request");
    let mut app = build_test_app(worker, queue, status);

    event_tx
        .send(WorkerEvent {
            request,
            result: Ok(Some(dummy_mesh())),
            elapsed: Duration::from_millis(1),
        })
        .expect("send worker event");

    app.update();

    let world = app.world_mut();
    let count = world.query::<&DraggableMesh>().iter(world).count();
    assert_eq!(count, 1);
}

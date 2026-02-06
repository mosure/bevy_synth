use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use bevy::prelude::*;
use bevy_mesh::Mesh as BevyMesh;
use burn_3d_synth_tripo::pipeline::mesh::Mesh as TripoMesh;

#[derive(Resource, Default)]
pub(crate) struct UiStatus {
    pub(crate) message: String,
    pub(crate) processing: bool,
}

#[derive(Resource)]
pub(crate) struct TitlePulse {
    pub(crate) timer: Timer,
    pub(crate) phase: usize,
}

impl Default for TitlePulse {
    fn default() -> Self {
        Self {
            timer: Timer::from_seconds(0.5, TimerMode::Repeating),
            phase: 0,
        }
    }
}

#[derive(Resource, Default)]
pub(crate) struct DragState {
    pub(crate) active: Option<DragSelection>,
}

#[derive(Resource, Default)]
pub(crate) struct ExitState {
    pub(crate) requested: bool,
}

pub(crate) struct DragSelection {
    pub(crate) entity: Entity,
    pub(crate) plane_y: f32,
    pub(crate) offset: Vec3,
}

#[derive(Resource, Default)]
pub(crate) struct InferenceQueue {
    pub(crate) active: Option<InferenceRequest>,
    pub(crate) pending: VecDeque<InferenceRequest>,
    pub(crate) counter: u32,
    pub(crate) completed: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct InferenceRequest {
    pub(crate) image_path: PathBuf,
    pub(crate) output_path: Option<PathBuf>,
}

#[derive(Resource)]
pub(crate) struct InferenceWorker {
    pub(crate) sender: Sender<WorkerCommand>,
    pub(crate) receiver: Mutex<Receiver<WorkerEvent>>,
}

pub(crate) enum WorkerCommand {
    Infer(InferenceRequest),
    Shutdown,
}

pub(crate) struct WorkerEvent {
    pub(crate) request: InferenceRequest,
    pub(crate) result: Result<Option<TripoMesh>, String>,
    pub(crate) elapsed: Duration,
}

#[derive(Component)]
pub(crate) struct DraggableMesh {
    pub(crate) local_min: Vec3,
    pub(crate) local_max: Vec3,
}

#[derive(Component)]
pub(crate) struct PendingAabb {
    pub(crate) handle: Handle<BevyMesh>,
}

#[derive(Component)]
pub(crate) struct Spinner;

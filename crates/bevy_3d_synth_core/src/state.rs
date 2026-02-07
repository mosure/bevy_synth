use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use bevy::prelude::*;
use burn_3d_synth_tripo::pipeline::mesh::Mesh as TripoMesh;

#[derive(Resource, Default)]
pub struct UiStatus {
    pub message: String,
    pub processing: bool,
    pub worker_message: Option<String>,
}

#[derive(Resource)]
pub struct TitlePulse {
    pub timer: Timer,
    pub phase: usize,
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
pub struct ExitState {
    pub requested: bool,
}

#[derive(Resource, Default)]
pub struct InferenceQueue {
    pub active: Option<Vec<InferenceRequest>>,
    pub pending: VecDeque<InferenceRequest>,
    pub counter: u32,
    pub completed: usize,
}

#[derive(Clone, Debug)]
pub struct InferenceRequest {
    pub id: u32,
    pub image_path: PathBuf,
    pub output_path: Option<PathBuf>,
}

#[derive(Resource)]
pub struct InferenceWorker {
    pub sender: Sender<WorkerCommand>,
    pub receiver: Mutex<Receiver<WorkerEvent>>,
}

pub enum WorkerCommand {
    Infer(Vec<InferenceRequest>),
    Shutdown,
}

pub struct WorkerEvent {
    pub requests: Vec<InferenceRequest>,
    pub results: Vec<Result<Option<TripoMesh>, String>>,
    pub elapsed: Duration,
    pub status_message: Option<String>,
}

#[derive(Component)]
pub struct Spinner;

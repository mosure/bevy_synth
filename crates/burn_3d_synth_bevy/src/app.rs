use std::path::PathBuf;
use std::sync::mpsc::TryRecvError;

use bevy::app::AppExit;
use bevy::camera::primitives::MeshAabb;
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::ecs::system::SystemParam;
use bevy::light::PointLight;
use bevy::pbr::{MeshMaterial3d, StandardMaterial};
use bevy::prelude::*;
use bevy::window::{FileDragAndDrop, PrimaryWindow, WindowCloseRequested};
use bevy_infinite_grid::{InfiniteGridBundle, InfiniteGridPlugin};
use bevy_mesh::{Mesh as BevyMesh, Mesh3d};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};
use clap::Parser;

use burn_3d_synth_tripo::pipeline::mesh::Mesh as TripoMesh;

use crate::args::{AppArgs, Args, build_app_args};
use crate::geom::{aabb_min_max, ray_aabb_intersection, ray_plane_intersection, world_aabb};
use crate::io::{is_image_file, is_mesh_file, resolve_output_path, write_obj};
use crate::mesh::{mesh_bounds, to_bevy_mesh};
use crate::state::{
    DragSelection, DragState, DraggableMesh, ExitState, InferenceQueue, InferenceRequest,
    InferenceWorker, PendingAabb, Spinner, TitlePulse, UiStatus, WorkerCommand,
};
use crate::worker::start_worker;

#[derive(SystemParam)]
struct FileDropContext<'w, 's> {
    events: MessageReader<'w, 's, FileDragAndDrop>,
    queue: ResMut<'w, InferenceQueue>,
    args: Res<'w, AppArgs>,
    asset_server: Res<'w, AssetServer>,
    commands: Commands<'w, 's>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    status: ResMut<'w, UiStatus>,
    exit_state: Res<'w, ExitState>,
}

#[derive(SystemParam)]
pub(crate) struct InferenceContext<'w, 's> {
    commands: Commands<'w, 's>,
    queue: ResMut<'w, InferenceQueue>,
    worker: Res<'w, InferenceWorker>,
    args: Res<'w, AppArgs>,
    meshes: ResMut<'w, Assets<BevyMesh>>,
    materials: ResMut<'w, Assets<StandardMaterial>>,
    status: ResMut<'w, UiStatus>,
    exit_state: Res<'w, ExitState>,
}

pub(crate) fn run() {
    if std::env::var("RUST_MIN_STACK").is_err() {
        unsafe {
            std::env::set_var("RUST_MIN_STACK", "67108864");
        }
    }
    let args = Args::parse();
    let app_args = build_app_args(args);

    let status_message = if app_args.image.is_none() && app_args.mesh.is_none() {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Initializing TripoSG viewer…".to_string()
    };

    App::new()
        .insert_resource(app_args)
        .insert_resource(InferenceQueue::default())
        .insert_resource(DragState::default())
        .insert_resource(ExitState::default())
        .insert_resource(TitlePulse::default())
        .insert_resource(UiStatus {
            message: status_message,
            processing: false,
        })
        .add_plugins(DefaultPlugins)
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(InfiniteGridPlugin)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                handle_exit_requests,
                handle_file_drop,
                drive_inference,
                update_pending_aabb,
                (drag_start, drag_update, drag_end).chain(),
                sync_panorbit_enabled,
                update_spinner,
                rotate_spinner,
                update_window_title,
            ),
        )
        .run();
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    args: Res<AppArgs>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
) {
    info!("burn_3d_synth_bevy args: {:?}", *args);

    commands.spawn((
        Camera3d::default(),
        Transform::from_translation(Vec3::new(0.0, 1.5, 5.0)),
        PanOrbitCamera {
            allow_upside_down: true,
            orbit_smoothness: 0.1,
            pan_smoothness: 0.1,
            zoom_smoothness: 0.1,
            ..default()
        },
    ));
    commands.spawn((PointLight::default(), Transform::from_xyz(2.0, 3.0, 2.0)));

    commands.spawn(InfiniteGridBundle::default());

    let worker = start_worker(args.as_ref());
    commands.insert_resource(worker);

    if let Some(mesh_path) = args.mesh.as_ref() {
        if mesh_path.exists() {
            spawn_mesh_asset(
                &mut commands,
                &asset_server,
                &mut materials,
                mesh_path.clone(),
            );
        } else {
            warn!("Mesh path {:?} does not exist; skipping", mesh_path);
        }
    }

    if let Some(image_path) = args.image.as_ref() {
        enqueue_inference(image_path.clone(), &args, &mut queue);
    }

    update_status_message(&args, &queue, &mut status);
}

fn handle_file_drop(mut ctx: FileDropContext) {
    if ctx.exit_state.requested {
        return;
    }
    for event in ctx.events.read() {
        let FileDragAndDrop::DroppedFile { path_buf, .. } = event else {
            continue;
        };

        if is_image_file(path_buf) {
            enqueue_inference(path_buf.clone(), &ctx.args, &mut ctx.queue);
            info!("Queued inference for {}", path_buf.display());
            continue;
        }

        if is_mesh_file(path_buf) {
            spawn_mesh_asset(
                &mut ctx.commands,
                &ctx.asset_server,
                &mut ctx.materials,
                path_buf.clone(),
            );
            info!("Loaded mesh asset {}", path_buf.display());
            continue;
        }

        warn!(
            "Dropped file {} is not a supported image or mesh",
            path_buf.display()
        );
    }

    update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
}

pub(crate) fn drive_inference(mut ctx: InferenceContext) {
    if ctx.exit_state.requested {
        return;
    }
    let receiver = match ctx.worker.receiver.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    loop {
        match receiver.try_recv() {
            Ok(event) => {
                let request = event.request.clone();
                ctx.queue.active = None;
                ctx.queue.completed += 1;
                info!(
                    "Inference completed in {:.2}s for {}",
                    event.elapsed.as_secs_f32(),
                    request.image_path.display()
                );
                handle_inference_result(
                    &mut ctx.commands,
                    &mut ctx.meshes,
                    &mut ctx.materials,
                    request,
                    event.result,
                );
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                warn!("Inference worker disconnected; pending requests will be skipped.");
                ctx.queue.active = None;
                ctx.queue.pending.clear();
                break;
            }
        }
    }

    if ctx.queue.active.is_none()
        && let Some(request) = ctx.queue.pending.pop_front()
    {
        if let Err(err) = ctx.worker.sender.send(WorkerCommand::Infer(request.clone())) {
            warn!("Failed to send inference request: {err}");
        } else {
            ctx.queue.active = Some(request);
        }
    }

    update_status_message(&ctx.args, &ctx.queue, &mut ctx.status);
}

fn update_pending_aabb(
    mut commands: Commands,
    meshes: Res<Assets<BevyMesh>>,
    pending: Query<(Entity, &PendingAabb)>,
) {
    for (entity, pending) in pending.iter() {
        let Some(mesh) = meshes.get(&pending.handle) else {
            continue;
        };
        let Some(aabb) = mesh.compute_aabb() else {
            continue;
        };
        let (min, max) = aabb_min_max(&aabb);
        commands
            .entity(entity)
            .insert(DraggableMesh {
                local_min: min,
                local_max: max,
            })
            .remove::<PendingAabb>();
    }
}

fn drag_start(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    draggables: Query<(Entity, &DraggableMesh, &GlobalTransform)>,
    mut drag_state: ResMut<DragState>,
) {
    if !buttons.just_pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };

    let mut best: Option<(Entity, f32, Vec3, Vec3)> = None;
    for (entity, bounds, transform) in draggables.iter() {
        let (min, max) = world_aabb(bounds, transform);
        let Some(dist) = ray_aabb_intersection(ray, min, max) else {
            continue;
        };
        let hit = ray.get_point(dist);
        let translation = transform.translation();
        if best
            .as_ref()
            .map(|(_, best_dist, _, _)| dist < *best_dist)
            .unwrap_or(true)
        {
            best = Some((entity, dist, hit, translation));
        }
    }

    if let Some((entity, _dist, hit, translation)) = best {
        let offset = translation - hit;
        drag_state.active = Some(DragSelection {
            entity,
            plane_y: translation.y,
            offset,
        });
    }
}

fn drag_update(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut drag_state: ResMut<DragState>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(selection) = drag_state.active.as_mut() else {
        return;
    };
    if !buttons.pressed(MouseButton::Left) {
        return;
    }

    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor) = window.cursor_position() else {
        return;
    };
    let Ok((camera, camera_transform)) = cameras.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_transform, cursor) else {
        return;
    };
    let Some(hit) = ray_plane_intersection(ray, selection.plane_y) else {
        return;
    };

    if let Ok(mut transform) = transforms.get_mut(selection.entity) {
        transform.translation = hit + selection.offset;
    }
}

fn drag_end(buttons: Res<ButtonInput<MouseButton>>, mut drag_state: ResMut<DragState>) {
    if buttons.just_released(MouseButton::Left) {
        drag_state.active = None;
    }
}

fn sync_panorbit_enabled(drag_state: Res<DragState>, mut cameras: Query<&mut PanOrbitCamera>) {
    let enabled = drag_state.active.is_none();
    for mut camera in cameras.iter_mut() {
        camera.enabled = enabled;
    }
}

fn update_spinner(queue: Res<InferenceQueue>, mut query: Query<&mut Visibility, With<Spinner>>) {
    let visible = queue.active.is_some();
    for mut visibility in query.iter_mut() {
        *visibility = if visible {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
}

fn rotate_spinner(
    time: Res<Time>,
    queue: Res<InferenceQueue>,
    mut query: Query<&mut Transform, With<Spinner>>,
) {
    if queue.active.is_none() {
        return;
    }
    for mut transform in query.iter_mut() {
        transform.rotate_y(time.delta_secs() * 1.5);
        transform.rotate_x(time.delta_secs() * 0.8);
    }
}

fn update_window_title(
    time: Res<Time>,
    queue: Res<InferenceQueue>,
    status: Res<UiStatus>,
    mut pulse: ResMut<TitlePulse>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let processing = queue.active.is_some();
    let mut should_update = status.is_changed();

    if processing {
        pulse.timer.tick(time.delta());
        if pulse.timer.just_finished() {
            pulse.phase = (pulse.phase + 1) % 4;
            should_update = true;
        }
    } else if pulse.phase != 0 {
        pulse.phase = 0;
        should_update = true;
    }

    if !should_update {
        return;
    }

    let title = if let Some(active) = queue.active.as_ref() {
        let dots = ".".repeat(pulse.phase);
        let name = active
            .image_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("image");
        format!(
            "TripoSG Viewer — Processing: {name} (Queued: {}){dots}",
            queue.pending.len()
        )
    } else {
        format!("TripoSG Viewer — {}", status.message)
    };

    if let Ok(mut window) = windows.single_mut() {
        window.title = title;
    }
}

fn handle_exit_requests(
    keys: Res<ButtonInput<KeyCode>>,
    mut close_events: MessageReader<WindowCloseRequested>,
    mut exit: MessageWriter<AppExit>,
    mut exit_state: ResMut<ExitState>,
    mut queue: ResMut<InferenceQueue>,
    mut status: ResMut<UiStatus>,
    worker: Res<InferenceWorker>,
) {
    if exit_state.requested {
        return;
    }
    let mut requested = keys.just_pressed(KeyCode::Escape);
    if close_events.read().next().is_some() {
        requested = true;
    }
    if !requested {
        return;
    }

    exit_state.requested = true;
    queue.active = None;
    queue.pending.clear();
    status.processing = false;
    status.message = "Shutting down…".to_string();
    info!("{}", status.message);
    let _ = worker.sender.send(WorkerCommand::Shutdown);
    exit.write(AppExit::Success);
}

fn handle_inference_result(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<BevyMesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    request: InferenceRequest,
    result: Result<Option<TripoMesh>, String>,
) {
    match result {
        Ok(Some(mesh)) => {
            if let Some(output) = request.output_path.as_ref()
                && let Err(err) = write_obj(output, &mesh)
            {
                warn!("failed to write mesh to {}: {err}", output.display());
            }

            let (local_min, local_max) = mesh_bounds(&mesh);
            let bevy_mesh = to_bevy_mesh(&mesh);
            let mesh_handle = meshes.add(bevy_mesh);
            let material = materials.add(StandardMaterial {
                base_color: Color::srgb(0.78, 0.84, 0.92),
                ..default()
            });

            commands.spawn((
                DraggableMesh {
                    local_min,
                    local_max,
                },
                Mesh3d(mesh_handle),
                MeshMaterial3d(material),
            ));
        }
        Ok(None) => {
            warn!(
                "TripoSG inference produced an empty mesh for {}",
                request.image_path.display()
            );
        }
        Err(err) => {
            warn!(
                "TripoSG inference failed for {}: {}",
                request.image_path.display(),
                err
            );
        }
    }
}

pub(crate) fn enqueue_inference(
    image_path: PathBuf,
    args: &AppArgs,
    queue: &mut InferenceQueue,
) {
    let output_path = resolve_output_path(args.output.as_ref(), &image_path, queue.counter);
    let request = InferenceRequest {
        image_path,
        output_path,
    };
    queue.counter = queue.counter.wrapping_add(1);
    queue.pending.push_back(request);
}

fn update_status_message(args: &AppArgs, queue: &InferenceQueue, status: &mut UiStatus) {
    let message = if let Some(active) = queue.active.as_ref() {
        format!(
            "Processing {} ({} queued)",
            active.image_path.display(),
            queue.pending.len()
        )
    } else if !queue.pending.is_empty() {
        format!("Queued {} inference job(s)…", queue.pending.len())
    } else if args.image.is_none() && args.mesh.is_none() && queue.completed == 0 {
        "Drag & drop an image (.png/.jpg) or mesh (.glb/.gltf/.obj) to begin.".to_string()
    } else {
        "Ready. Drag & drop another image or mesh to add more.".to_string()
    };

    status.processing = queue.active.is_some();
    if status.message != message {
        status.message = message;
        info!("{}", status.message);
    }
}

fn spawn_mesh_asset(
    commands: &mut Commands,
    asset_server: &Res<AssetServer>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
    mesh_path: PathBuf,
) {
    let mesh_handle: Handle<BevyMesh> = asset_server.load(mesh_path);
    let material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.82, 0.9),
        ..default()
    });
    commands.spawn((
        PendingAabb {
            handle: mesh_handle.clone(),
        },
        Mesh3d(mesh_handle),
        MeshMaterial3d(material),
    ));
}

use super::*;

#[cfg(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda")))]
pub(super) fn worker_loop_backend_unavailable(
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
    message: &'static str,
) {
    for command in command_rx {
        match command {
            WorkerCommand::Infer(requests) => {
                let results = vec![Err(message.to_string()); requests.len()];
                let _ = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: std::time::Duration::ZERO,
                    status_message: None,
                });
            }
            WorkerCommand::Shutdown => break,
        }
    }
}

#[cfg(not(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda"))))]
#[allow(dead_code)]
pub(super) fn worker_loop_backend_unavailable(
    _command_rx: Receiver<WorkerCommand>,
    _event_tx: Sender<WorkerEvent>,
    _message: &'static str,
) {
    unreachable!("worker_loop_backend_unavailable is unreachable when all native backends exist");
}

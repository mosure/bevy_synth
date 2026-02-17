use super::*;

#[cfg(any(target_arch = "wasm32", not(feature = "wgpu"), not(feature = "cuda")))]
pub(super) fn worker_loop_backend_unavailable(
    command_rx: Receiver<WorkerCommand>,
    event_tx: Sender<WorkerEvent>,
    message: &'static str,
    #[cfg(not(target_arch = "wasm32"))] wake_callback: Option<WorkerWakeCallback>,
) {
    for command in command_rx {
        match command {
            WorkerCommand::Infer(requests) => {
                let results = vec![Err(message.to_string()); requests.len()];
                let sent = event_tx.send(WorkerEvent {
                    requests,
                    results,
                    elapsed: std::time::Duration::ZERO,
                    status_message: None,
                });
                #[cfg(not(target_arch = "wasm32"))]
                if sent.is_ok()
                    && let Some(wake) = wake_callback.as_ref()
                {
                    wake();
                }
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
    #[cfg(not(target_arch = "wasm32"))] _wake_callback: Option<WorkerWakeCallback>,
) {
    unreachable!("worker_loop_backend_unavailable is unreachable when all native backends exist");
}

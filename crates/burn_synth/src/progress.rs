use std::fmt::{Debug, Formatter};
use std::sync::Arc;

/// Controls how much runtime progress is emitted.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum ProgressVerbosity {
    #[default]
    Off,
    Stages,
    Steps,
}

/// Structured runtime progress event emitted by `SynthRuntime`.
#[derive(Clone, Debug)]
pub enum RuntimeProgressEvent {
    RunStarted {
        run: &'static str,
        detail: Option<String>,
    },
    StageStarted {
        run: &'static str,
        stage: &'static str,
        total_steps: Option<usize>,
        detail: Option<String>,
    },
    Step {
        run: &'static str,
        stage: &'static str,
        step: usize,
        total_steps: usize,
        step_ms: f64,
        elapsed_ms: f64,
        eta_ms: Option<f64>,
        detail: Option<String>,
    },
    StageCompleted {
        run: &'static str,
        stage: &'static str,
        total_steps: Option<usize>,
        elapsed_ms: f64,
        detail: Option<String>,
    },
    Warning {
        run: &'static str,
        message: String,
    },
    RunCompleted {
        run: &'static str,
        elapsed_ms: f64,
        detail: Option<String>,
    },
}

pub type ProgressCallback = Arc<dyn Fn(&RuntimeProgressEvent) + Send + Sync + 'static>;

/// Shared observer config used by runtime call sites (CLI, MCP, Bevy).
#[derive(Clone)]
pub struct RuntimeProgressObserver {
    pub verbosity: ProgressVerbosity,
    pub step_interval: usize,
    callback: Option<ProgressCallback>,
}

impl Default for RuntimeProgressObserver {
    fn default() -> Self {
        Self {
            verbosity: ProgressVerbosity::Off,
            step_interval: 1,
            callback: None,
        }
    }
}

impl Debug for RuntimeProgressObserver {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeProgressObserver")
            .field("verbosity", &self.verbosity)
            .field("step_interval", &self.step_interval)
            .field("has_callback", &self.callback.is_some())
            .finish()
    }
}

impl RuntimeProgressObserver {
    pub fn with_callback(
        verbosity: ProgressVerbosity,
        step_interval: usize,
        callback: ProgressCallback,
    ) -> Self {
        Self {
            verbosity,
            step_interval: step_interval.max(1),
            callback: Some(callback),
        }
    }

    pub fn callback(&self) -> Option<&ProgressCallback> {
        self.callback.as_ref()
    }

    pub fn set_callback(&mut self, callback: Option<ProgressCallback>) {
        self.callback = callback;
    }

    pub fn is_enabled(&self) -> bool {
        self.callback.is_some() && !matches!(self.verbosity, ProgressVerbosity::Off)
    }

    pub fn emits_stages(&self) -> bool {
        self.callback.is_some() && !matches!(self.verbosity, ProgressVerbosity::Off)
    }

    pub fn emits_steps(&self) -> bool {
        self.callback.is_some() && matches!(self.verbosity, ProgressVerbosity::Steps)
    }

    pub fn should_emit_step(&self, step: usize, total_steps: usize) -> bool {
        if !self.emits_steps() {
            return false;
        }
        let interval = self.step_interval.max(1);
        step == 1 || step == total_steps || step.is_multiple_of(interval)
    }

    pub fn emit(&self, event: RuntimeProgressEvent) {
        if let Some(callback) = self.callback.as_ref() {
            callback(&event);
        }
    }
}

pub fn default_log_progress_callback() -> ProgressCallback {
    Arc::new(log_progress_event)
}

pub fn log_progress_event(event: &RuntimeProgressEvent) {
    match event {
        RuntimeProgressEvent::RunStarted { run, detail } => {
            if let Some(detail) = detail {
                log::info!("burn_synth.progress run={run} status=started detail=\"{detail}\"");
            } else {
                log::info!("burn_synth.progress run={run} status=started");
            }
        }
        RuntimeProgressEvent::StageStarted {
            run,
            stage,
            total_steps,
            detail,
        } => {
            if let Some(detail) = detail {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} status=started total_steps={} detail=\"{detail}\"",
                    total_steps.map_or_else(|| "-".to_string(), |v| v.to_string())
                );
            } else {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} status=started total_steps={}",
                    total_steps.map_or_else(|| "-".to_string(), |v| v.to_string())
                );
            }
        }
        RuntimeProgressEvent::Step {
            run,
            stage,
            step,
            total_steps,
            step_ms,
            elapsed_ms,
            eta_ms,
            detail,
        } => {
            let percent = if *total_steps > 0 {
                (*step as f64 / *total_steps as f64) * 100.0
            } else {
                0.0
            };
            if let Some(detail) = detail {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} step={step}/{total_steps} progress={percent:.1}% step_ms={step_ms:.1} elapsed_ms={elapsed_ms:.1} eta_ms={} detail=\"{detail}\"",
                    eta_ms.map_or_else(|| "-".to_string(), |value| format!("{value:.1}"))
                );
            } else {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} step={step}/{total_steps} progress={percent:.1}% step_ms={step_ms:.1} elapsed_ms={elapsed_ms:.1} eta_ms={}",
                    eta_ms.map_or_else(|| "-".to_string(), |value| format!("{value:.1}"))
                );
            }
        }
        RuntimeProgressEvent::StageCompleted {
            run,
            stage,
            total_steps,
            elapsed_ms,
            detail,
        } => {
            if let Some(detail) = detail {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} status=completed total_steps={} elapsed_ms={elapsed_ms:.1} detail=\"{detail}\"",
                    total_steps.map_or_else(|| "-".to_string(), |v| v.to_string())
                );
            } else {
                log::info!(
                    "burn_synth.progress run={run} stage={stage} status=completed total_steps={} elapsed_ms={elapsed_ms:.1}",
                    total_steps.map_or_else(|| "-".to_string(), |v| v.to_string())
                );
            }
        }
        RuntimeProgressEvent::Warning { run, message } => {
            log::warn!("burn_synth.progress run={run} status=warning message=\"{message}\"");
        }
        RuntimeProgressEvent::RunCompleted {
            run,
            elapsed_ms,
            detail,
        } => {
            if let Some(detail) = detail {
                log::info!(
                    "burn_synth.progress run={run} status=completed elapsed_ms={elapsed_ms:.1} detail=\"{detail}\""
                );
            } else {
                log::info!(
                    "burn_synth.progress run={run} status=completed elapsed_ms={elapsed_ms:.1}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ProgressVerbosity, RuntimeProgressObserver};

    #[test]
    fn step_sampling_respects_interval_and_boundaries() {
        let observer = RuntimeProgressObserver {
            verbosity: ProgressVerbosity::Steps,
            step_interval: 3,
            callback: Some(Arc::new(|_| {})),
            ..RuntimeProgressObserver::default()
        };
        assert!(!observer.should_emit_step(2, 10));
        assert!(observer.should_emit_step(1, 10));
        assert!(observer.should_emit_step(3, 10));
        assert!(observer.should_emit_step(10, 10));
    }

    #[test]
    fn stages_mode_disables_step_events() {
        let observer = RuntimeProgressObserver {
            verbosity: ProgressVerbosity::Stages,
            step_interval: 1,
            callback: Some(Arc::new(|_| {})),
            ..RuntimeProgressObserver::default()
        };
        assert!(!observer.should_emit_step(1, 10));
    }
}

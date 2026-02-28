use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeModelDebugConfig {
    pub stage_debug: bool,
    pub attention_debug: bool,
}

static STAGE_DEBUG: AtomicBool = AtomicBool::new(false);
static ATTENTION_DEBUG: AtomicBool = AtomicBool::new(false);

pub fn set_runtime_model_debug_config(config: RuntimeModelDebugConfig) {
    STAGE_DEBUG.store(config.stage_debug, Ordering::Relaxed);
    ATTENTION_DEBUG.store(config.attention_debug, Ordering::Relaxed);
}

pub fn runtime_model_stage_debug_enabled() -> bool {
    STAGE_DEBUG.load(Ordering::Relaxed)
}

pub fn runtime_model_attention_debug_enabled() -> bool {
    ATTENTION_DEBUG.load(Ordering::Relaxed)
}

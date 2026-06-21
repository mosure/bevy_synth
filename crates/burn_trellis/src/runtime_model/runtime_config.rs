use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, Debug, Default)]
pub struct RuntimeModelDebugConfig {
    pub stage_debug: bool,
    pub attention_debug: bool,
    pub sparse_flow_module_attention: bool,
    pub sparse_flow_module_attention_f16: bool,
    pub sparse_flow_linear_f16: bool,
    pub sparse_flow_torso_f16: bool,
    pub sparse_flow_coord_rope_kernel: bool,
    pub sparse_decoder_conv_f16: bool,
}

static STAGE_DEBUG: AtomicBool = AtomicBool::new(false);
static ATTENTION_DEBUG: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_MODULE_ATTENTION: AtomicBool = AtomicBool::new(true);
static SPARSE_FLOW_MODULE_ATTENTION_F16: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_LINEAR_F16: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_TORSO_F16: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_COORD_ROPE_KERNEL: AtomicBool = AtomicBool::new(true);
static SPARSE_DECODER_CONV_F16: AtomicBool = AtomicBool::new(false);

pub fn set_runtime_model_debug_config(config: RuntimeModelDebugConfig) {
    STAGE_DEBUG.store(config.stage_debug, Ordering::Relaxed);
    ATTENTION_DEBUG.store(config.attention_debug, Ordering::Relaxed);
    SPARSE_FLOW_MODULE_ATTENTION.store(config.sparse_flow_module_attention, Ordering::Relaxed);
    SPARSE_FLOW_MODULE_ATTENTION_F16
        .store(config.sparse_flow_module_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_LINEAR_F16.store(config.sparse_flow_linear_f16, Ordering::Relaxed);
    SPARSE_FLOW_TORSO_F16.store(config.sparse_flow_torso_f16, Ordering::Relaxed);
    SPARSE_FLOW_COORD_ROPE_KERNEL.store(config.sparse_flow_coord_rope_kernel, Ordering::Relaxed);
    SPARSE_DECODER_CONV_F16.store(config.sparse_decoder_conv_f16, Ordering::Relaxed);
}

pub fn runtime_model_stage_debug_enabled() -> bool {
    STAGE_DEBUG.load(Ordering::Relaxed)
}

pub fn runtime_model_attention_debug_enabled() -> bool {
    ATTENTION_DEBUG.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_flow_module_attention_enabled() -> bool {
    SPARSE_FLOW_MODULE_ATTENTION.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_flow_module_attention_f16_enabled() -> bool {
    SPARSE_FLOW_MODULE_ATTENTION_F16.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_flow_linear_f16_enabled() -> bool {
    SPARSE_FLOW_LINEAR_F16.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_flow_torso_f16_enabled() -> bool {
    SPARSE_FLOW_TORSO_F16.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_flow_coord_rope_kernel_enabled() -> bool {
    SPARSE_FLOW_COORD_ROPE_KERNEL.load(Ordering::Relaxed)
}

pub fn runtime_model_sparse_decoder_conv_f16_enabled() -> bool {
    SPARSE_DECODER_CONV_F16.load(Ordering::Relaxed)
}

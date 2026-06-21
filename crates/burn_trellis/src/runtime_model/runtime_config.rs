use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
static SPARSE_FLOW_SELF_MODULE_ATTENTION_F16: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_CROSS_MODULE_ATTENTION_F16: AtomicBool = AtomicBool::new(false);
static SPARSE_FLOW_FINAL_F32_STEPS: AtomicUsize = AtomicUsize::new(0);
static SPARSE_FLOW_CURRENT_STEP: AtomicUsize = AtomicUsize::new(usize::MAX);
static SPARSE_FLOW_CURRENT_STEP_COUNT: AtomicUsize = AtomicUsize::new(0);
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
    SPARSE_FLOW_SELF_MODULE_ATTENTION_F16
        .store(config.sparse_flow_module_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_CROSS_MODULE_ATTENTION_F16
        .store(config.sparse_flow_module_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_FINAL_F32_STEPS.store(0, Ordering::Relaxed);
    clear_runtime_model_sparse_flow_sampler_step();
    SPARSE_FLOW_LINEAR_F16.store(config.sparse_flow_linear_f16, Ordering::Relaxed);
    SPARSE_FLOW_TORSO_F16.store(config.sparse_flow_torso_f16, Ordering::Relaxed);
    SPARSE_FLOW_COORD_ROPE_KERNEL.store(config.sparse_flow_coord_rope_kernel, Ordering::Relaxed);
    SPARSE_DECODER_CONV_F16.store(config.sparse_decoder_conv_f16, Ordering::Relaxed);
}

pub fn set_runtime_model_sparse_flow_attention_policy(
    self_attention_f16: bool,
    cross_attention_f16: bool,
    final_f32_steps: usize,
) {
    SPARSE_FLOW_SELF_MODULE_ATTENTION_F16.store(self_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_CROSS_MODULE_ATTENTION_F16.store(cross_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_MODULE_ATTENTION_F16
        .store(self_attention_f16 || cross_attention_f16, Ordering::Relaxed);
    SPARSE_FLOW_FINAL_F32_STEPS.store(final_f32_steps, Ordering::Relaxed);
    clear_runtime_model_sparse_flow_sampler_step();
}

pub fn set_runtime_model_sparse_flow_sampler_step(step_idx: usize, step_count: usize) {
    SPARSE_FLOW_CURRENT_STEP.store(step_idx, Ordering::Relaxed);
    SPARSE_FLOW_CURRENT_STEP_COUNT.store(step_count, Ordering::Relaxed);
}

pub fn clear_runtime_model_sparse_flow_sampler_step() {
    SPARSE_FLOW_CURRENT_STEP.store(usize::MAX, Ordering::Relaxed);
    SPARSE_FLOW_CURRENT_STEP_COUNT.store(0, Ordering::Relaxed);
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
    runtime_model_sparse_flow_self_attention_f16_enabled()
        || runtime_model_sparse_flow_cross_attention_f16_enabled()
}

pub fn runtime_model_sparse_flow_self_attention_f16_enabled() -> bool {
    sparse_flow_attention_f16_enabled_for_current_step(
        SPARSE_FLOW_SELF_MODULE_ATTENTION_F16.load(Ordering::Relaxed),
    )
}

pub fn runtime_model_sparse_flow_cross_attention_f16_enabled() -> bool {
    sparse_flow_attention_f16_enabled_for_current_step(
        SPARSE_FLOW_CROSS_MODULE_ATTENTION_F16.load(Ordering::Relaxed),
    )
}

fn sparse_flow_attention_f16_enabled_for_current_step(base_enabled: bool) -> bool {
    if !base_enabled {
        return false;
    }
    let final_f32_steps = SPARSE_FLOW_FINAL_F32_STEPS.load(Ordering::Relaxed);
    if final_f32_steps == 0 {
        return true;
    }
    let step_idx = SPARSE_FLOW_CURRENT_STEP.load(Ordering::Relaxed);
    let step_count = SPARSE_FLOW_CURRENT_STEP_COUNT.load(Ordering::Relaxed);
    if step_idx == usize::MAX || step_count == 0 {
        return true;
    }
    step_idx.saturating_add(final_f32_steps) < step_count
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

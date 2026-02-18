#![recursion_limit = "256"]

pub mod model;
pub mod paths;
pub mod pipeline;
pub(crate) mod readback;
#[cfg(target_arch = "wasm32")]
mod wasm_meshopt_alloc;

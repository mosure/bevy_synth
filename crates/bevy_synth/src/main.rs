#![recursion_limit = "256"]

mod app;
mod infinite_grid;
#[cfg(all(
    target_arch = "wasm32",
    target_os = "unknown",
    not(feature = "triposg")
))]
mod wasm_cpp_alloc;

#[cfg(test)]
mod tests;

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    app::run();
}

#![recursion_limit = "256"]

mod app;

#[cfg(test)]
mod tests;

fn main() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();

    app::run();
}

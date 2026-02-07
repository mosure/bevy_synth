#![recursion_limit = "256"]

mod app;

#[cfg(test)]
mod tests;

fn main() {
    app::run();
}

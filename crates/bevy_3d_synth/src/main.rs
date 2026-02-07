#![recursion_limit = "256"]

mod app;
mod ui;

#[cfg(test)]
mod tests;

fn main() {
    app::run();
}

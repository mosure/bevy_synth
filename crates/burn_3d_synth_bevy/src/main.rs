#![recursion_limit = "256"]

mod app;
mod args;
mod geom;
mod io;
mod mesh;
mod paths;
mod state;
mod worker;

#[cfg(test)]
mod tests;

fn main() {
    app::run();
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
pub struct Instant(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl Instant {
    pub fn now() -> Self {
        Self(std::time::Instant::now())
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.0.elapsed()
    }
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy, Debug)]
pub struct Instant {
    started_ms: f64,
}

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Self {
            started_ms: js_sys::Date::now(),
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        let ms = (js_sys::Date::now() - self.started_ms).max(0.0);
        std::time::Duration::from_secs_f64(ms / 1000.0)
    }
}

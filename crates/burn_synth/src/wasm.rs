/// Canonical wasm inference preset for JS-facing API entry points.
///
/// These values intentionally target stable browser execution for smoke testing while
/// still exercising the real TripoSG + RMBG path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmInferencePreset {
    pub quality: &'static str,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub resolution: usize,
    pub faces: usize,
    pub backend: &'static str,
    pub rmbg_backend: &'static str,
    pub dino_backend: &'static str,
}

impl Default for WasmInferencePreset {
    fn default() -> Self {
        Self {
            quality: "fast",
            num_steps: 8,
            num_tokens: 512,
            resolution: 128,
            faces: 5000,
            backend: "wgpu",
            rmbg_backend: "auto",
            dino_backend: "auto",
        }
    }
}

impl WasmInferencePreset {
    /// Build CLI-style args consumed by runtime argument parsing.
    pub fn to_cli_args(&self, program_name: &str) -> Vec<String> {
        vec![
            program_name.to_string(),
            "--quality".to_string(),
            self.quality.to_string(),
            "--num-steps".to_string(),
            self.num_steps.to_string(),
            "--num-tokens".to_string(),
            self.num_tokens.to_string(),
            "--resolution".to_string(),
            self.resolution.to_string(),
            "--faces".to_string(),
            self.faces.to_string(),
            "--backend".to_string(),
            self.backend.to_string(),
            "--rmbg-backend".to_string(),
            self.rmbg_backend.to_string(),
            "--dino-backend".to_string(),
            self.dino_backend.to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::WasmInferencePreset;

    #[test]
    fn preset_generates_expected_args() {
        let args = WasmInferencePreset::default().to_cli_args("bevy_synth");
        assert_eq!(
            args,
            vec![
                "bevy_synth",
                "--quality",
                "fast",
                "--num-steps",
                "8",
                "--num-tokens",
                "512",
                "--resolution",
                "128",
                "--faces",
                "5000",
                "--backend",
                "wgpu",
                "--rmbg-backend",
                "auto",
                "--dino-backend",
                "auto",
            ]
        );
    }
}

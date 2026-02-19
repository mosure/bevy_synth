/// Canonical wasm inference preset for JS-facing API entry points.
///
/// Defaults mirror native TripoSG "full" quality settings so web and native runs
/// are configured consistently unless callers override fields explicitly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmInferencePreset {
    pub quality: &'static str,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub resolution: usize,
    pub faces: usize,
    pub seed: u64,
    pub backend: &'static str,
    pub rmbg_backend: &'static str,
    pub dino_backend: &'static str,
    pub weights_precision: &'static str,
    pub rmbg_weights_precision: &'static str,
}

impl Default for WasmInferencePreset {
    fn default() -> Self {
        Self {
            quality: "full",
            num_steps: 50,
            num_tokens: 2048,
            // On wasm this maps to flash extraction min_resolution.
            resolution: 63,
            faces: 10_000,
            seed: 42,
            backend: "wgpu",
            rmbg_backend: "auto",
            dino_backend: "auto",
            weights_precision: "f16",
            rmbg_weights_precision: "auto",
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
            "--seed".to_string(),
            self.seed.to_string(),
            "--backend".to_string(),
            self.backend.to_string(),
            "--rmbg-backend".to_string(),
            self.rmbg_backend.to_string(),
            "--dino-backend".to_string(),
            self.dino_backend.to_string(),
            "--weights-precision".to_string(),
            self.weights_precision.to_string(),
            "--rmbg-weights-precision".to_string(),
            self.rmbg_weights_precision.to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::WasmInferencePreset;
    #[cfg(feature = "runtime")]
    use crate::RuntimeConfig;

    #[test]
    fn preset_generates_expected_args() {
        let args = WasmInferencePreset::default().to_cli_args("bevy_synth");
        assert_eq!(
            args,
            vec![
                "bevy_synth",
                "--quality",
                "full",
                "--num-steps",
                "50",
                "--num-tokens",
                "2048",
                "--resolution",
                "63",
                "--faces",
                "10000",
                "--seed",
                "42",
                "--backend",
                "wgpu",
                "--rmbg-backend",
                "auto",
                "--dino-backend",
                "auto",
                "--weights-precision",
                "f16",
                "--rmbg-weights-precision",
                "auto",
            ]
        );
    }

    #[cfg(feature = "runtime")]
    #[test]
    fn preset_matches_runtime_triposg_defaults() {
        let preset = WasmInferencePreset::default();
        let runtime = RuntimeConfig::default();

        assert_eq!(preset.num_steps, runtime.num_steps);
        assert_eq!(preset.num_tokens, runtime.num_tokens);
        assert_eq!(preset.resolution, runtime.flash_extract.min_resolution);
        assert_eq!(preset.faces, runtime.target_faces.unwrap_or_default());
        assert_eq!(Some(preset.seed), runtime.seed);
    }
}

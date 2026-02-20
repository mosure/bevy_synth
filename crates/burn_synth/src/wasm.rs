/// Canonical wasm inference preset for JS-facing API entry points.
///
/// Defaults mirror the CLI "balanced" quality preset so web and native runs
/// are configured consistently unless callers override fields explicitly.
pub const DEFAULT_WASM_FLASH_NUM_CHUNKS: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmInferencePreset {
    pub quality: &'static str,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub resolution: usize,
    pub faces: usize,
    pub flash_octree_depth: usize,
    pub flash_num_chunks: usize,
    pub flash_mini_grid_num: usize,
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
            quality: "balanced",
            num_steps: 20,
            num_tokens: 1024,
            // On wasm this maps to flash extraction min_resolution.
            resolution: 31,
            faces: 10_000,
            flash_octree_depth: 8,
            // Keep wasm flash chunking conservative for broader WebGPU portability
            // (notably Metal/f16 adapters) while preserving output parity.
            flash_num_chunks: DEFAULT_WASM_FLASH_NUM_CHUNKS,
            flash_mini_grid_num: 4,
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
            "--flash-octree-depth".to_string(),
            self.flash_octree_depth.to_string(),
            "--flash-num-chunks".to_string(),
            self.flash_num_chunks.to_string(),
            "--flash-mini-grid-num".to_string(),
            self.flash_mini_grid_num.to_string(),
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
    use super::{DEFAULT_WASM_FLASH_NUM_CHUNKS, WasmInferencePreset};
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
                "balanced",
                "--num-steps",
                "20",
                "--num-tokens",
                "1024",
                "--resolution",
                "31",
                "--faces",
                "10000",
                "--flash-octree-depth",
                "8",
                "--flash-num-chunks",
                "4096",
                "--flash-mini-grid-num",
                "4",
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
    fn preset_defaults_to_balanced_quality_values() {
        let preset = WasmInferencePreset::default();
        let runtime = RuntimeConfig::default();

        assert_eq!(preset.quality, "balanced");
        assert_eq!(preset.num_steps, 20);
        assert_eq!(preset.num_tokens, 1024);
        assert_eq!(preset.resolution, 31);
        assert_eq!(preset.flash_octree_depth, 8);
        assert_eq!(preset.flash_num_chunks, DEFAULT_WASM_FLASH_NUM_CHUNKS);
        assert_eq!(preset.flash_mini_grid_num, 4);
        assert_eq!(preset.faces, runtime.target_faces.unwrap_or_default());
        assert_eq!(Some(preset.seed), runtime.seed);
    }
}

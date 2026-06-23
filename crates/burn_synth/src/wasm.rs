/// Canonical wasm inference preset for JS-facing API entry points.
///
/// Defaults mirror the CLI "balanced" quality preset so web and native runs
/// are configured consistently unless callers override fields explicitly.
pub const DEFAULT_WASM_FLASH_NUM_CHUNKS: usize = 4096;

use crate::quality::{DEFAULT_SEED, DEFAULT_TRIPOSG_TARGET_FACES, RuntimeQualityPreset};

#[derive(Clone, Debug, PartialEq)]
pub struct WasmInferencePreset {
    pub quality: &'static str,
    pub synthesis_model: &'static str,
    pub rmbg_model: &'static str,
    pub num_steps: usize,
    pub num_tokens: usize,
    pub guidance_scale: f32,
    pub triposplat_shift: f32,
    pub triposplat_num_gaussians: usize,
    pub triposplat_erode_radius: usize,
    pub resolution: usize,
    pub faces: usize,
    pub trellis_max_sparse_coords: usize,
    pub trellis_pbr_enabled: bool,
    pub trellis_pbr_texture_size: usize,
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
        let quality = RuntimeQualityPreset::Balanced.defaults();
        let triposplat = burn_triposplat::TripoSplatProfile::Balanced.settings();
        Self {
            quality: "balanced",
            synthesis_model: "triposg",
            rmbg_model: "rmbg14",
            num_steps: quality.num_steps,
            num_tokens: quality.num_tokens,
            guidance_scale: quality.guidance_scale,
            triposplat_shift: burn_triposplat::DEFAULT_SHIFT,
            triposplat_num_gaussians: triposplat.num_gaussians,
            triposplat_erode_radius: burn_triposplat::DEFAULT_ERODE_RADIUS,
            // On wasm this maps to flash extraction min_resolution.
            resolution: 31,
            faces: DEFAULT_TRIPOSG_TARGET_FACES,
            trellis_max_sparse_coords: 2_048,
            trellis_pbr_enabled: false,
            trellis_pbr_texture_size: 1024,
            flash_octree_depth: 8,
            // Keep wasm flash chunking conservative for broader WebGPU portability
            // (notably Metal/f16 adapters) while preserving output parity.
            flash_num_chunks: DEFAULT_WASM_FLASH_NUM_CHUNKS,
            flash_mini_grid_num: 4,
            seed: DEFAULT_SEED,
            backend: "wgpu",
            rmbg_backend: "auto",
            dino_backend: "auto",
            weights_precision: "auto",
            rmbg_weights_precision: "auto",
        }
    }
}

impl WasmInferencePreset {
    pub fn triposplat_default() -> Self {
        let settings = burn_triposplat::TripoSplatProfile::Balanced.settings();
        let mut preset = Self::default();
        preset.synthesis_model = "triposplat";
        preset.num_steps = settings.steps;
        preset.guidance_scale = settings.guidance_scale;
        preset.triposplat_num_gaussians = settings.num_gaussians;
        preset
    }

    /// Build CLI-style args consumed by runtime argument parsing.
    pub fn to_cli_args(&self, program_name: &str) -> Vec<String> {
        let mut args = vec![
            program_name.to_string(),
            "--quality".to_string(),
            self.quality.to_string(),
            "--synthesis-models".to_string(),
            self.synthesis_model.to_string(),
            "--rmbg-model".to_string(),
            self.rmbg_model.to_string(),
            "--num-steps".to_string(),
            self.num_steps.to_string(),
            "--num-tokens".to_string(),
            self.num_tokens.to_string(),
            "--guidance-scale".to_string(),
            self.guidance_scale.to_string(),
            "--triposplat-shift".to_string(),
            self.triposplat_shift.to_string(),
            "--gaussians".to_string(),
            self.triposplat_num_gaussians.to_string(),
            "--triposplat-erode-radius".to_string(),
            self.triposplat_erode_radius.to_string(),
            "--resolution".to_string(),
            self.resolution.to_string(),
            "--faces".to_string(),
            self.faces.to_string(),
            "--trellis-max-sparse-coords".to_string(),
            self.trellis_max_sparse_coords.to_string(),
            "--trellis-pbr-texture-size".to_string(),
            self.trellis_pbr_texture_size.to_string(),
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
        ];
        args.extend([
            "--trellis-pbr".to_string(),
            self.trellis_pbr_enabled.to_string(),
        ]);
        args
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "runtime")]
    use super::DEFAULT_WASM_FLASH_NUM_CHUNKS;
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
                "balanced",
                "--synthesis-models",
                "triposg",
                "--rmbg-model",
                "rmbg14",
                "--num-steps",
                "20",
                "--num-tokens",
                "1024",
                "--guidance-scale",
                "7",
                "--triposplat-shift",
                "3",
                "--gaussians",
                "262144",
                "--triposplat-erode-radius",
                "1",
                "--resolution",
                "31",
                "--faces",
                "10000",
                "--trellis-max-sparse-coords",
                "2048",
                "--trellis-pbr-texture-size",
                "1024",
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
                "auto",
                "--rmbg-weights-precision",
                "auto",
                "--trellis-pbr",
                "false",
            ]
        );

        let mut pbr_preset = WasmInferencePreset::default();
        pbr_preset.trellis_pbr_enabled = true;
        let pbr_args = pbr_preset.to_cli_args("bevy_synth");
        assert!(pbr_args.ends_with(&["--trellis-pbr".to_string(), "true".to_string()]));
    }

    #[cfg(feature = "runtime")]
    #[test]
    fn preset_defaults_to_balanced_quality_values() {
        let preset = WasmInferencePreset::default();
        let runtime = RuntimeConfig::default();

        assert_eq!(preset.quality, "balanced");
        assert_eq!(preset.synthesis_model, "triposg");
        assert_eq!(preset.rmbg_model, "rmbg14");
        assert_eq!(preset.num_steps, 20);
        assert_eq!(preset.num_tokens, 1024);
        assert_eq!(preset.guidance_scale, 7.0);
        assert_eq!(preset.triposplat_shift, 3.0);
        assert_eq!(preset.triposplat_num_gaussians, 262_144);
        assert_eq!(preset.triposplat_erode_radius, 1);
        assert_eq!(preset.resolution, 31);
        assert!(!preset.trellis_pbr_enabled);
        assert_eq!(preset.trellis_max_sparse_coords, 2048);
        assert_eq!(preset.trellis_pbr_texture_size, 1024);
        assert_eq!(preset.flash_octree_depth, 8);
        assert_eq!(preset.flash_num_chunks, DEFAULT_WASM_FLASH_NUM_CHUNKS);
        assert_eq!(preset.flash_mini_grid_num, 4);
        assert_eq!(preset.faces, runtime.target_faces.unwrap_or_default());
        assert_eq!(Some(preset.seed), runtime.seed);
    }

    #[test]
    fn triposplat_default_uses_pipeline_specific_guidance() {
        let preset = WasmInferencePreset::triposplat_default();
        assert_eq!(preset.synthesis_model, "triposplat");
        assert_eq!(preset.num_steps, burn_triposplat::DEFAULT_NUM_STEPS);
        assert_eq!(
            preset.guidance_scale,
            burn_triposplat::DEFAULT_GUIDANCE_SCALE
        );
        assert_eq!(
            preset.triposplat_num_gaussians,
            burn_triposplat::DEFAULT_NUM_GAUSSIANS
        );
    }
}

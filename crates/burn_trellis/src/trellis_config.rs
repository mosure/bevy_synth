use std::collections::BTreeMap;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisPipelineConfig {
    #[serde(default = "default_pipeline_name")]
    pub name: String,
    pub args: TrellisPipelineArgs,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisPipelineArgs {
    #[serde(default)]
    pub models: BTreeMap<String, String>,
    #[serde(default = "default_sparse_sampler")]
    pub sparse_structure_sampler: TrellisSamplerConfig,
    #[serde(default = "default_shape_sampler")]
    pub shape_slat_sampler: TrellisSamplerConfig,
    #[serde(default = "default_tex_sampler")]
    pub tex_slat_sampler: TrellisSamplerConfig,
    #[serde(default = "default_shape_normalization")]
    pub shape_slat_normalization: TrellisNormalization,
    #[serde(default = "default_tex_normalization")]
    pub tex_slat_normalization: TrellisNormalization,
    #[serde(default = "default_pipeline_type")]
    pub default_pipeline_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisSamplerConfig {
    #[serde(default = "default_sampler_name")]
    pub name: String,
    #[serde(default)]
    pub args: TrellisSamplerArgs,
    #[serde(default)]
    pub params: TrellisSamplerParams,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisSamplerArgs {
    #[serde(default = "default_sigma_min")]
    pub sigma_min: f32,
}

impl Default for TrellisSamplerArgs {
    fn default() -> Self {
        Self {
            sigma_min: default_sigma_min(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisSamplerParams {
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_guidance_strength")]
    pub guidance_strength: f32,
    #[serde(default)]
    pub guidance_rescale: f32,
    #[serde(default = "default_guidance_interval")]
    pub guidance_interval: [f32; 2],
    #[serde(default = "default_rescale_t")]
    pub rescale_t: f32,
}

impl Default for TrellisSamplerParams {
    fn default() -> Self {
        Self {
            steps: default_steps(),
            guidance_strength: default_guidance_strength(),
            guidance_rescale: 0.0,
            guidance_interval: default_guidance_interval(),
            rescale_t: default_rescale_t(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrellisNormalization {
    #[serde(default = "default_norm_channels")]
    pub mean: Vec<f32>,
    #[serde(default = "default_norm_channels")]
    pub std: Vec<f32>,
}

impl TrellisPipelineConfig {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }
}

fn default_pipeline_type() -> String {
    "1024_cascade".to_string()
}

fn default_pipeline_name() -> String {
    "Trellis2ImageTo3DPipeline".to_string()
}

fn default_sampler_name() -> String {
    "FlowEulerGuidanceIntervalSampler".to_string()
}

fn default_sigma_min() -> f32 {
    1.0e-5
}

fn default_steps() -> usize {
    12
}

fn default_guidance_strength() -> f32 {
    1.0
}

fn default_guidance_interval() -> [f32; 2] {
    [0.0, 1.0]
}

fn default_rescale_t() -> f32 {
    1.0
}

fn default_norm_channels() -> Vec<f32> {
    vec![0.0; 32]
}

fn default_sampler() -> TrellisSamplerConfig {
    TrellisSamplerConfig {
        name: default_sampler_name(),
        args: TrellisSamplerArgs {
            sigma_min: default_sigma_min(),
        },
        params: TrellisSamplerParams {
            steps: default_steps(),
            guidance_strength: default_guidance_strength(),
            guidance_rescale: 0.0,
            guidance_interval: default_guidance_interval(),
            rescale_t: default_rescale_t(),
        },
    }
}

fn default_sparse_sampler() -> TrellisSamplerConfig {
    let mut sampler = default_sampler();
    sampler.params.guidance_strength = 7.5;
    sampler.params.guidance_rescale = 0.7;
    sampler.params.guidance_interval = [0.6, 1.0];
    sampler.params.rescale_t = 5.0;
    sampler
}

fn default_shape_sampler() -> TrellisSamplerConfig {
    let mut sampler = default_sampler();
    sampler.params.guidance_strength = 7.5;
    sampler.params.guidance_rescale = 0.5;
    sampler.params.guidance_interval = [0.6, 1.0];
    sampler.params.rescale_t = 3.0;
    sampler
}

fn default_tex_sampler() -> TrellisSamplerConfig {
    let mut sampler = default_sampler();
    sampler.params.guidance_strength = 1.0;
    sampler.params.guidance_rescale = 0.0;
    sampler.params.guidance_interval = [0.6, 0.9];
    sampler.params.rescale_t = 3.0;
    sampler
}

fn default_shape_normalization() -> TrellisNormalization {
    TrellisNormalization {
        mean: vec![0.0; 32],
        std: vec![1.0; 32],
    }
}

fn default_tex_normalization() -> TrellisNormalization {
    TrellisNormalization {
        mean: vec![0.0; 32],
        std: vec![1.0; 32],
    }
}

#[cfg(test)]
mod tests {
    use super::TrellisPipelineConfig;

    #[test]
    fn parses_pipeline_json() {
        let json = br#"{
            "name": "Trellis2ImageTo3DPipeline",
            "args": {
                "models": { "shape": "ckpts/shape" },
                "sparse_structure_sampler": {
                    "name": "FlowEulerGuidanceIntervalSampler",
                    "args": { "sigma_min": 1e-5 },
                    "params": {
                        "steps": 12,
                        "guidance_strength": 7.5,
                        "guidance_rescale": 0.7,
                        "guidance_interval": [0.6, 1.0],
                        "rescale_t": 5.0
                    }
                },
                "shape_slat_sampler": {
                    "name": "FlowEulerGuidanceIntervalSampler",
                    "args": { "sigma_min": 1e-5 },
                    "params": {
                        "steps": 12,
                        "guidance_strength": 7.5,
                        "guidance_rescale": 0.5,
                        "guidance_interval": [0.6, 1.0],
                        "rescale_t": 3.0
                    }
                },
                "tex_slat_sampler": {
                    "name": "FlowEulerGuidanceIntervalSampler",
                    "args": { "sigma_min": 1e-5 },
                    "params": {
                        "steps": 12,
                        "guidance_strength": 1.0,
                        "guidance_rescale": 0.0,
                        "guidance_interval": [0.6, 0.9],
                        "rescale_t": 3.0
                    }
                },
                "shape_slat_normalization": { "mean": [0.0, 1.0], "std": [1.0, 2.0] },
                "tex_slat_normalization": { "mean": [0.0], "std": [1.0] },
                "default_pipeline_type": "1024_cascade"
            }
        }"#;
        let parsed = TrellisPipelineConfig::from_json_bytes(json).expect("json should parse");
        assert_eq!(parsed.name, "Trellis2ImageTo3DPipeline");
        assert_eq!(parsed.args.default_pipeline_type, "1024_cascade");
        assert_eq!(parsed.args.sparse_structure_sampler.params.steps, 12);
        assert!(parsed.args.models.contains_key("shape"));
    }
}

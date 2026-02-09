#![cfg_attr(not(feature = "runtime-model"), allow(dead_code))]

use std::collections::HashMap;
#[cfg(feature = "runtime-model")]
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use crate::mesh::{Mesh, MeshMaterial, MeshPbrTextures, MeshTexture};
use crate::preprocess::PreprocessOutput;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::fdg_decoder::FdgDecoderRuntime;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_decoder::SparseSubdivisionLogits;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_structure_flow::SparseStructureFlowRuntime;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_unet_vae_decoder::SparseUnetVaeDecoderRuntime;
use crate::sampler::{FlowEulerGuidanceIntervalSampler, FlowEulerSampleConfig};
use crate::trellis_config::{TrellisNormalization, TrellisPipelineArgs, TrellisSamplerConfig};

#[path = "staged_pipeline_decode.rs"]
mod staged_pipeline_decode;
use staged_pipeline_decode::*;

#[derive(Debug, Clone)]
pub struct SparseStructureSample {
    pub source: SparseStructureStageSource,
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub resolution: usize,
    pub flow_resolution: usize,
    pub flow_channels: usize,
    pub noise: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
    pub step_mid_x_t: Vec<f32>,
    pub step_last_x_t: Vec<f32>,
    pub latent: Vec<f32>,
    pub coords: Vec<[u32; 4]>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SparseStructureStageSource {
    Synthetic,
    RuntimeModelCpu,
    RuntimeModelWgpu,
}

impl SparseStructureStageSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::RuntimeModelCpu => "runtime_model_cpu",
            Self::RuntimeModelWgpu => "runtime_model_wgpu",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShapeSLatSample {
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub dense_resolution: usize,
    pub dense_channels: usize,
    pub dense_noise: Option<Vec<f32>>,
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_mid_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TexSLatSample {
    pub sampler_config: FlowEulerSampleConfig,
    pub sigma_min: f32,
    pub step_count: usize,
    pub dense_resolution: usize,
    pub dense_channels: usize,
    pub dense_noise: Option<Vec<f32>>,
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_mid_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub shape_slat_cond: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TrellisStageOutput {
    pub sparse: SparseStructureSample,
    pub shape_slat: ShapeSLatSample,
    pub tex_slat: TexSLatSample,
    pub decode_shape_subs: Vec<DecodeShapeSubSample>,
    pub decode_tex_voxels: DecodeTexVoxelSample,
    pub mesh: Mesh,
    pub pbr: Option<PbrBakeDebug>,
}

#[derive(Debug, Clone)]
pub struct DecodeShapeSubSample {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 8]>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
pub struct DecodeTexVoxelSample {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 6]>,
    pub spatial_shape: [u32; 3],
}

#[derive(Debug, Clone)]
struct DecodedLatentOutput {
    mesh: Mesh,
    shape_subs: Vec<DecodeShapeSubSample>,
    tex_voxels: DecodeTexVoxelSample,
    pbr: Option<PbrBakeDebug>,
}

#[derive(Debug, Clone)]
pub struct PbrBakeDebug {
    pub texture_width: usize,
    pub texture_height: usize,
    pub uvs: Vec<[f32; 2]>,
    pub raster_mask: Vec<u8>,
    pub sample_positions: Vec<[f32; 3]>,
    pub sample_attrs: Vec<[f32; 6]>,
    pub base_color_float: Vec<[f32; 4]>,
    pub metallic_float: Vec<f32>,
    pub roughness_float: Vec<f32>,
    pub alpha_float: Vec<f32>,
    pub base_color_rgba_u8: Vec<u8>,
    pub metallic_roughness_u8: Vec<u8>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrellisStageTimings {
    pub sparse_ms: f64,
    pub shape_slat_ms: f64,
    pub tex_slat_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SparseRowNoiseOverride {
    pub coords: Vec<[u32; 4]>,
    pub feats: Vec<[f32; 32]>,
}

#[derive(Debug, Clone, Default)]
pub struct TrellisNoiseOverrides {
    pub sparse_noise: Option<Vec<f32>>,
    pub sparse_coords: Option<Vec<[u32; 4]>>,
    pub shape_noise: Option<SparseRowNoiseOverride>,
    pub tex_noise: Option<SparseRowNoiseOverride>,
    pub shape_noise_dense: Option<Vec<f32>>,
    pub tex_noise_dense: Option<Vec<f32>>,
    pub sparse_sampler: Option<SamplerConfigOverride>,
    pub shape_sampler: Option<SamplerConfigOverride>,
    pub tex_sampler: Option<SamplerConfigOverride>,
    pub cond_512: Option<Vec<f32>>,
    pub neg_cond_512: Option<Vec<f32>>,
    pub cond_1024: Option<Vec<f32>>,
    pub neg_cond_1024: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
pub struct SamplerConfigOverride {
    pub sigma_min: f32,
    pub config: FlowEulerSampleConfig,
}

impl TrellisNoiseOverrides {
    pub fn is_empty(&self) -> bool {
        self.sparse_noise.is_none()
            && self.sparse_coords.is_none()
            && self.shape_noise.is_none()
            && self.tex_noise.is_none()
            && self.shape_noise_dense.is_none()
            && self.tex_noise_dense.is_none()
            && self.sparse_sampler.is_none()
            && self.shape_sampler.is_none()
            && self.tex_sampler.is_none()
            && self.cond_512.is_none()
            && self.neg_cond_512.is_none()
            && self.cond_1024.is_none()
            && self.neg_cond_1024.is_none()
    }
}

#[derive(Debug)]
pub struct TrellisStageRuntime {
    pipeline_type: String,
    sparse_sampler: TrellisSamplerConfig,
    shape_sampler: TrellisSamplerConfig,
    tex_sampler: TrellisSamplerConfig,
    shape_norm: TrellisNormalization,
    tex_norm: TrellisNormalization,
    #[cfg(feature = "runtime-model")]
    sparse_flow: Option<SparseStructureFlowRuntime>,
    #[cfg(feature = "runtime-model")]
    shape_flow: Option<SparseStructureFlowRuntime>,
    #[cfg(feature = "runtime-model")]
    tex_flow: Option<SparseStructureFlowRuntime>,
    #[cfg(feature = "runtime-model")]
    shape_decoder: Option<FdgDecoderRuntime>,
    #[cfg(feature = "runtime-model")]
    tex_decoder: Option<SparseUnetVaeDecoderRuntime>,
}

impl TrellisStageRuntime {
    pub fn from_args(args: &TrellisPipelineArgs, preferred_pipeline_type: Option<&str>) -> Self {
        Self::from_args_with_assets(args, preferred_pipeline_type, None, None, false)
    }

    pub fn from_args_with_assets(
        args: &TrellisPipelineArgs,
        preferred_pipeline_type: Option<&str>,
        _weights_root: Option<&Path>,
        _image_large_root: Option<&Path>,
        _prefer_wgpu: bool,
    ) -> Self {
        let pipeline_type = preferred_pipeline_type
            .unwrap_or(args.default_pipeline_type.as_str())
            .to_string();
        let mut sparse_sampler = args.sparse_structure_sampler.clone();
        let mut shape_sampler = args.shape_slat_sampler.clone();
        let mut tex_sampler = args.tex_slat_sampler.clone();
        if let Some(steps_override) = runtime_sampler_steps_override() {
            sparse_sampler.params.steps = steps_override;
            shape_sampler.params.steps = steps_override;
            tex_sampler.params.steps = steps_override;
            eprintln!("burn_trellis: sampler steps override active (steps={steps_override})");
        }
        #[cfg(feature = "runtime-model")]
        let runtime_model_disabled = std::env::var("TRELLIS2_DISABLE_RUNTIME_MODEL")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        #[cfg(feature = "runtime-model")]
        let runtime_decoders_disabled = std::env::var("TRELLIS2_DISABLE_RUNTIME_DECODERS")
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false);
        #[cfg(feature = "runtime-model")]
        let slat_dense_resolution = std::env::var("TRELLIS2_SLAT_DENSE_RESOLUTION")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0);
        #[cfg(feature = "runtime-model")]
        let prefer_512_slat = matches!(pipeline_type.as_str(), "512" | "512_base");
        #[cfg(feature = "runtime-model")]
        let shape_flow_key = if prefer_512_slat {
            "shape_slat_flow_model_512"
        } else {
            "shape_slat_flow_model_1024"
        };
        #[cfg(feature = "runtime-model")]
        let tex_flow_key = if prefer_512_slat {
            "tex_slat_flow_model_512"
        } else {
            "tex_slat_flow_model_1024"
        };
        #[cfg(feature = "runtime-model")]
        let sparse_flow = if runtime_model_disabled {
            None
        } else {
            match (
                _weights_root,
                args.models.get("sparse_structure_flow_model"),
            ) {
                (Some(weights_root), Some(model_stem)) => {
                    match SparseStructureFlowRuntime::load_from_stem(
                        weights_root,
                        _image_large_root,
                        model_stem,
                        _prefer_wgpu,
                        None,
                    ) {
                        Ok(runtime) => {
                            eprintln!(
                                "burn_trellis: sparse flow runtime backend = {}",
                                runtime.backend_name()
                            );
                            Some(runtime)
                        }
                        Err(err) => {
                            eprintln!(
                                "burn_trellis: sparse flow runtime model unavailable ({err}); using synthetic sparse stage fallback."
                            );
                            None
                        }
                    }
                }
                _ => None,
            }
        };
        #[cfg(feature = "runtime-model")]
        let shape_flow = if runtime_model_disabled {
            None
        } else {
            match (_weights_root, args.models.get(shape_flow_key)) {
                (Some(weights_root), Some(model_stem)) => {
                    match SparseStructureFlowRuntime::load_from_stem(
                        weights_root,
                        _image_large_root,
                        model_stem,
                        _prefer_wgpu,
                        slat_dense_resolution,
                    ) {
                        Ok(runtime) => {
                            eprintln!(
                                "burn_trellis: shape slat runtime backend = {} (flow={}, dense_res={})",
                                runtime.backend_name(),
                                shape_flow_key,
                                runtime.config().resolution
                            );
                            Some(runtime)
                        }
                        Err(err) => {
                            eprintln!(
                                "burn_trellis: shape slat runtime model unavailable for key '{}' ({err}); using synthetic shape stage fallback.",
                                shape_flow_key
                            );
                            None
                        }
                    }
                }
                _ => None,
            }
        };
        #[cfg(feature = "runtime-model")]
        let tex_flow = if runtime_model_disabled {
            None
        } else {
            match (_weights_root, args.models.get(tex_flow_key)) {
                (Some(weights_root), Some(model_stem)) => {
                    match SparseStructureFlowRuntime::load_from_stem(
                        weights_root,
                        _image_large_root,
                        model_stem,
                        _prefer_wgpu,
                        slat_dense_resolution,
                    ) {
                        Ok(runtime) => {
                            eprintln!(
                                "burn_trellis: tex slat runtime backend = {} (flow={}, dense_res={})",
                                runtime.backend_name(),
                                tex_flow_key,
                                runtime.config().resolution
                            );
                            Some(runtime)
                        }
                        Err(err) => {
                            eprintln!(
                                "burn_trellis: tex slat runtime model unavailable for key '{}' ({err}); using synthetic tex stage fallback.",
                                tex_flow_key
                            );
                            None
                        }
                    }
                }
                _ => None,
            }
        };
        #[cfg(feature = "runtime-model")]
        let shape_decoder = if runtime_model_disabled || runtime_decoders_disabled {
            if runtime_decoders_disabled {
                eprintln!(
                    "burn_trellis: runtime decoders disabled by TRELLIS2_DISABLE_RUNTIME_DECODERS."
                );
            }
            None
        } else {
            match (_weights_root, args.models.get("shape_slat_decoder")) {
                (Some(weights_root), Some(model_stem)) => {
                    match FdgDecoderRuntime::load_from_stem(
                        weights_root,
                        _image_large_root,
                        model_stem,
                        _prefer_wgpu,
                    ) {
                        Ok(runtime) => Some(runtime),
                        Err(err) => {
                            eprintln!(
                                "burn_trellis: shape decoder runtime unavailable ({err}); decode stage will fail until runtime decoder assets are available."
                            );
                            None
                        }
                    }
                }
                _ => None,
            }
        };
        #[cfg(feature = "runtime-model")]
        let tex_decoder = if runtime_model_disabled || runtime_decoders_disabled {
            None
        } else {
            match (_weights_root, args.models.get("tex_slat_decoder")) {
                (Some(weights_root), Some(model_stem)) => {
                    match SparseUnetVaeDecoderRuntime::load_from_stem(
                        weights_root,
                        _image_large_root,
                        model_stem,
                        _prefer_wgpu,
                    ) {
                        Ok(runtime) => Some(runtime),
                        Err(err) => {
                            eprintln!(
                                "burn_trellis: tex decoder runtime unavailable ({err}); decode stage will fail until runtime decoder assets are available."
                            );
                            None
                        }
                    }
                }
                _ => None,
            }
        };
        Self {
            pipeline_type,
            sparse_sampler,
            shape_sampler,
            tex_sampler,
            shape_norm: args.shape_slat_normalization.clone(),
            tex_norm: args.tex_slat_normalization.clone(),
            #[cfg(feature = "runtime-model")]
            sparse_flow,
            #[cfg(feature = "runtime-model")]
            shape_flow,
            #[cfg(feature = "runtime-model")]
            tex_flow,
            #[cfg(feature = "runtime-model")]
            shape_decoder,
            #[cfg(feature = "runtime-model")]
            tex_decoder,
        }
    }

    pub fn pipeline_type(&self) -> &str {
        self.pipeline_type.as_str()
    }

    pub fn run(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> Result<TrellisStageOutput, String> {
        self.run_with_overrides(preprocess, seed, None)
    }

    pub fn run_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
    ) -> Result<TrellisStageOutput, String> {
        self.run_profiled_with_overrides(preprocess, seed, noise_overrides, false)
            .map(|(output, _timings)| output)
    }

    pub fn run_profiled(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> Result<(TrellisStageOutput, TrellisStageTimings), String> {
        self.run_profiled_with_overrides(preprocess, seed, None, false)
    }

    pub fn run_profiled_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
        capture_sampler_trace: bool,
    ) -> Result<(TrellisStageOutput, TrellisStageTimings), String> {
        let total_start = Instant::now();
        let stage_debug = runtime_stage_debug_enabled();
        let parity_strict = runtime_parity_strict();
        let sparse_resolution = sparse_resolution_for_pipeline(self.pipeline_type());
        let mut rng = Lcg::new(seed);
        let sparse_noise_override = noise_overrides.and_then(|v| v.sparse_noise.as_deref());
        let sparse_coords_override = noise_overrides.and_then(|v| v.sparse_coords.as_deref());
        let shape_noise_override = noise_overrides.and_then(|v| v.shape_noise.as_ref());
        let tex_noise_override = noise_overrides.and_then(|v| v.tex_noise.as_ref());
        let shape_noise_dense_override =
            noise_overrides.and_then(|v| v.shape_noise_dense.as_deref());
        let tex_noise_dense_override = noise_overrides.and_then(|v| v.tex_noise_dense.as_deref());
        let sparse_sampler_override = noise_overrides.and_then(|v| v.sparse_sampler);
        let shape_sampler_override = noise_overrides.and_then(|v| v.shape_sampler);
        let tex_sampler_override = noise_overrides.and_then(|v| v.tex_sampler);
        let sparse_cond_override = noise_overrides.and_then(|v| v.cond_512.as_deref());
        let sparse_neg_cond_override = noise_overrides.and_then(|v| v.neg_cond_512.as_deref());
        let sparse_start = Instant::now();
        let sparse = sample_sparse_structure(
            preprocess,
            sparse_resolution,
            &mut rng,
            sparse_noise_override,
            sparse_coords_override,
            sparse_cond_override,
            sparse_neg_cond_override,
            &self.sparse_sampler,
            sparse_sampler_override,
            capture_sampler_trace,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.sparse_flow.as_ref(),
        )?;
        let sparse_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;
        if stage_debug {
            eprintln!(
                "burn_trellis: stage sparse complete ({sparse_ms:.2} ms, coords={})",
                sparse.coords.len()
            );
        }

        let shape_start = Instant::now();
        let shape_slat = sample_shape_slat(
            preprocess,
            &sparse.coords,
            &mut rng,
            shape_noise_override,
            shape_noise_dense_override,
            noise_overrides,
            &self.shape_sampler,
            shape_sampler_override,
            &self.shape_norm,
            sparse.resolution,
            capture_sampler_trace,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.shape_flow.as_ref(),
        )?;
        let shape_slat_ms = shape_start.elapsed().as_secs_f64() * 1000.0;
        if stage_debug {
            eprintln!(
                "burn_trellis: stage shape_slat complete ({shape_slat_ms:.2} ms, rows={})",
                shape_slat.coords.len()
            );
        }

        let tex_start = Instant::now();
        let tex_slat = sample_tex_slat(
            preprocess,
            &shape_slat,
            &mut rng,
            tex_noise_override,
            tex_noise_dense_override,
            noise_overrides,
            &self.tex_sampler,
            tex_sampler_override,
            &self.shape_norm,
            &self.tex_norm,
            sparse.resolution,
            capture_sampler_trace,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.tex_flow.as_ref(),
        )?;
        let tex_slat_ms = tex_start.elapsed().as_secs_f64() * 1000.0;
        if stage_debug {
            eprintln!(
                "burn_trellis: stage tex_slat complete ({tex_slat_ms:.2} ms, rows={})",
                tex_slat.coords.len()
            );
        }

        let decode_start = Instant::now();
        let decoded = if runtime_skip_decode() {
            if parity_strict {
                return Err(
                    "burn_trellis parity strict mode: TRELLIS2_SKIP_DECODE cannot be used in parity mode"
                        .to_string(),
                );
            }
            DecodedLatentOutput {
                mesh: canonical_cube(),
                shape_subs: Vec::new(),
                tex_voxels: DecodeTexVoxelSample {
                    coords: Vec::new(),
                    feats: Vec::new(),
                    spatial_shape: [1, 1, 1],
                },
                pbr: None,
            }
        } else {
            decode_latent_to_outputs(
                &shape_slat,
                &tex_slat,
                self.pipeline_type(),
                parity_strict,
                #[cfg(feature = "runtime-model")]
                self.shape_decoder.as_ref(),
                #[cfg(feature = "runtime-model")]
                self.tex_decoder.as_ref(),
            )?
        };
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        if stage_debug {
            eprintln!(
                "burn_trellis: stage decode complete ({decode_ms:.2} ms, vertices={}, faces={})",
                decoded.mesh.vertices.len(),
                decoded.mesh.faces.len()
            );
        }
        let output = TrellisStageOutput {
            sparse,
            shape_slat,
            tex_slat,
            decode_shape_subs: decoded.shape_subs,
            decode_tex_voxels: decoded.tex_voxels,
            mesh: decoded.mesh,
            pbr: decoded.pbr,
        };
        let timings = TrellisStageTimings {
            sparse_ms,
            shape_slat_ms,
            tex_slat_ms,
            decode_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        };
        Ok((output, timings))
    }
}

fn runtime_sampler_steps_override() -> Option<usize> {
    std::env::var("TRELLIS2_SAMPLER_STEPS_OVERRIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn runtime_parity_strict() -> bool {
    for key in ["TRELLIS2_PARITY_STRICT", "TRELLIS2_E2E_STRICT"] {
        if let Ok(value) = std::env::var(key)
            && matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        {
            return true;
        }
    }
    false
}

fn runtime_stage_debug_enabled() -> bool {
    std::env::var("TRELLIS2_STAGE_DEBUG")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn runtime_skip_decode() -> bool {
    std::env::var("TRELLIS2_SKIP_DECODE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn resolve_sampler_settings(
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
) -> (FlowEulerGuidanceIntervalSampler, FlowEulerSampleConfig, f32) {
    if let Some(override_config) = sampler_override {
        let sampler = FlowEulerGuidanceIntervalSampler::new(override_config.sigma_min);
        return (sampler, override_config.config, override_config.sigma_min);
    }
    let sigma_min = sampler_config.args.sigma_min;
    let (sampler, config) =
        FlowEulerGuidanceIntervalSampler::from_params(sigma_min, &sampler_config.params);
    (sampler, config, sigma_min)
}

fn dense_noise_with_override(
    rng: &mut Lcg,
    expected_len: usize,
    override_values: Option<&[f32]>,
    stage: &str,
) -> Vec<f32> {
    if let Some(values) = override_values {
        if values.len() == expected_len {
            return values.to_vec();
        }
        eprintln!(
            "burn_trellis: ignoring {stage} noise override due to len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        );
    }
    (0..expected_len).map(|_| rng.next_normal_f32()).collect()
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn build_dense_runtime_noise(
    rng: &mut Lcg,
    channels: usize,
    voxel_count: usize,
    dense_override: Option<&[f32]>,
    sparse_row_override: Option<&SparseRowNoiseOverride>,
    active_coords: &[[u32; 4]],
    sparse_resolution: usize,
    dense_resolution: usize,
    stage: &str,
) -> Vec<f32> {
    let mut noise = dense_noise_with_override(
        rng,
        channels.saturating_mul(voxel_count),
        dense_override,
        stage,
    );
    if let Some(override_rows) = sparse_row_override {
        merge_sparse_row_noise_override(
            noise.as_mut_slice(),
            override_rows,
            active_coords,
            channels,
            sparse_resolution,
            dense_resolution,
            stage,
        );
    }
    noise
}

#[cfg(feature = "runtime-model")]
fn resize_override_values(values: &[f32], expected_len: usize) -> Option<Vec<f32>> {
    if expected_len == 0 {
        return Some(Vec::new());
    }
    if values.is_empty() {
        return None;
    }
    if values.len() == expected_len {
        return Some(values.to_vec());
    }
    if values.len() == 1 {
        return Some(vec![values[0]; expected_len]);
    }

    let src_last = values.len() - 1;
    let dst_last = expected_len - 1;
    let mut out = Vec::with_capacity(expected_len);
    for dst_idx in 0..expected_len {
        let src_pos = dst_idx as f64 * src_last as f64 / dst_last.max(1) as f64;
        let src_floor = src_pos.floor() as usize;
        let src_ceil = src_pos.ceil() as usize;
        if src_floor == src_ceil {
            out.push(values[src_floor]);
            continue;
        }
        let t = (src_pos - src_floor as f64) as f32;
        let a = values[src_floor];
        let b = values[src_ceil];
        out.push(a * (1.0 - t) + b * t);
    }
    Some(out)
}

#[cfg(feature = "runtime-model")]
fn cond_override_for_tokens(
    overrides: Option<&TrellisNoiseOverrides>,
    cond_tokens: usize,
) -> (Option<&[f32]>, Option<&[f32]>) {
    const TOKENS_512: usize = 32 * 32 + 5;
    const TOKENS_1024: usize = 64 * 64 + 5;
    let Some(overrides) = overrides else {
        return (None, None);
    };
    match cond_tokens {
        TOKENS_512 => (
            overrides.cond_512.as_deref(),
            overrides.neg_cond_512.as_deref(),
        ),
        TOKENS_1024 => (
            overrides.cond_1024.as_deref(),
            overrides.neg_cond_1024.as_deref(),
        ),
        _ => (None, None),
    }
}

#[cfg(feature = "runtime-model")]
fn dense_cond_with_override(
    preprocess: &PreprocessOutput,
    cond_tokens: usize,
    cond_channels: usize,
    override_values: Option<&[f32]>,
    stage: &str,
) -> Result<Vec<f32>, String> {
    let expected = cond_tokens.saturating_mul(cond_channels);
    if let Some(values) = override_values {
        if values.len() == expected {
            return Ok(values.to_vec());
        }
        if runtime_parity_strict() {
            return Err(format!(
                "strict mode rejects {stage} cond override len mismatch (expected {}, got {})",
                expected,
                values.len()
            ));
        }
        if let Some(resized) = resize_override_values(values, expected) {
            eprintln!(
                "burn_trellis: resized {stage} cond override from {} to {} values",
                values.len(),
                expected
            );
            return Ok(resized);
        }
        eprintln!(
            "burn_trellis: ignoring {stage} cond override due to len mismatch (expected {}, got {})",
            expected,
            values.len()
        );
    }
    Ok(build_sparse_cond_from_preprocess(
        preprocess,
        cond_tokens,
        cond_channels,
    ))
}

#[cfg(feature = "runtime-model")]
fn dense_neg_cond_with_override(
    expected_len: usize,
    override_values: Option<&[f32]>,
    stage: &str,
) -> Result<Vec<f32>, String> {
    if let Some(values) = override_values {
        if values.len() == expected_len {
            return Ok(values.to_vec());
        }
        if runtime_parity_strict() {
            return Err(format!(
                "strict mode rejects {stage} neg-cond override len mismatch (expected {}, got {})",
                expected_len,
                values.len()
            ));
        }
        if let Some(resized) = resize_override_values(values, expected_len) {
            eprintln!(
                "burn_trellis: resized {stage} neg-cond override from {} to {} values",
                values.len(),
                expected_len
            );
            return Ok(resized);
        }
        eprintln!(
            "burn_trellis: ignoring {stage} neg-cond override due to len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        );
    }
    Ok(vec![0.0; expected_len])
}

fn sparse_row_noise_map(override_rows: &SparseRowNoiseOverride) -> HashMap<u64, [f32; 32]> {
    let count = override_rows.coords.len().min(override_rows.feats.len());
    let mut out = HashMap::with_capacity(count * 2);
    for idx in 0..count {
        let coord = override_rows.coords[idx];
        out.insert(
            pack_coord(coord[1], coord[2], coord[3]),
            override_rows.feats[idx],
        );
    }
    out
}

#[cfg(feature = "runtime-model")]
fn merge_sparse_row_noise_override(
    dense_noise: &mut [f32],
    override_rows: &SparseRowNoiseOverride,
    active_coords: &[[u32; 4]],
    channels: usize,
    sparse_resolution: usize,
    dense_resolution: usize,
    stage: &str,
) {
    if channels == 0 || dense_noise.is_empty() {
        return;
    }
    let voxel_count = dense_noise.len() / channels.max(1);
    if voxel_count == 0 || dense_noise.len() != channels * voxel_count {
        return;
    }

    let active_keys: HashSet<u64> = active_coords
        .iter()
        .map(|coord| pack_coord(coord[1], coord[2], coord[3]))
        .collect();
    let count = override_rows.coords.len().min(override_rows.feats.len());
    let mut merged = 0usize;
    for idx in 0..count {
        let coord = override_rows.coords[idx];
        let key = pack_coord(coord[1], coord[2], coord[3]);
        if !active_keys.contains(&key) {
            continue;
        }
        let dense_idx = map_coord_to_dense_flat(coord, sparse_resolution, dense_resolution);
        if dense_idx >= voxel_count {
            continue;
        }
        let row = override_rows.feats[idx];
        for ch in 0..channels.min(32) {
            dense_noise[ch * voxel_count + dense_idx] = row[ch];
        }
        merged += 1;
    }
    if runtime_stage_debug_enabled() {
        eprintln!("burn_trellis: merged {merged} sparse-row noise overrides for stage {stage}");
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_sparse_structure(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    coords_override: Option<&[[u32; 4]]>,
    _cond_override: Option<&[f32]>,
    _neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    capture_sampler_trace: bool,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] sparse_flow: Option<&SparseStructureFlowRuntime>,
) -> Result<SparseStructureSample, String> {
    #[cfg(feature = "runtime-model")]
    if let Some(sparse_flow) = sparse_flow
        && let Some(sample) = sample_sparse_structure_with_model(
            preprocess,
            resolution,
            rng,
            noise_override,
            coords_override,
            _cond_override,
            _neg_cond_override,
            sampler_config,
            sampler_override,
            capture_sampler_trace,
            sparse_flow,
        )
    {
        return Ok(sample);
    }
    if parity_strict {
        return Err(
            "burn_trellis parity strict mode: sparse_structure stage would use synthetic fallback"
                .to_string(),
        );
    }
    Ok(sample_sparse_structure_synthetic(
        preprocess,
        resolution,
        rng,
        noise_override,
        coords_override,
        sampler_config,
        sampler_override,
        capture_sampler_trace,
    ))
}

#[allow(clippy::too_many_arguments)]
fn sample_sparse_structure_synthetic(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    coords_override: Option<&[[u32; 4]]>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    capture_sampler_trace: bool,
) -> SparseStructureSample {
    let flow_resolution = 16usize;
    let flow_channels = 8usize;
    let voxel_count = flow_resolution * flow_resolution * flow_resolution;
    let noise =
        dense_noise_with_override(rng, flow_channels * voxel_count, noise_override, "sparse");
    let target = occupancy_target(preprocess, flow_resolution);
    let (sampler, sample_cfg, sigma_min) =
        resolve_sampler_settings(sampler_config, sampler_override);
    let trace = sampler.sample_with_trace_mode(
        &noise,
        sample_cfg,
        capture_sampler_trace,
        |x_t, _t, cond| {
            // Placeholder denoiser: positive branch drifts toward the occupancy target,
            // negative branch drifts toward empty space.
            let mut out = vec![0.0f32; x_t.len()];
            for idx in 0..out.len() {
                let target_idx = idx % voxel_count;
                let target_value = if cond { target[target_idx] } else { 0.0 };
                out[idx] = x_t[idx] - target_value;
            }
            out
        },
    );
    let latent = trace.samples;
    let occupancy = latent_to_occupancy(&latent, flow_channels, flow_resolution);
    let upsampled = upsample_occupancy(occupancy.as_slice(), flow_resolution, resolution);
    let mut coords = if let Some(override_coords) = coords_override {
        override_coords.to_vec()
    } else {
        let mut sampled = Vec::new();
        let threshold = 0.5f32;
        for z in 0..resolution {
            for y in 0..resolution {
                for x in 0..resolution {
                    let flat = (z * resolution + y) * resolution + x;
                    if upsampled[flat] <= threshold {
                        continue;
                    }
                    sampled.push([0, x as u32, y as u32, z as u32]);
                }
            }
        }
        sampled
    };
    if coords.is_empty() {
        coords.push([
            0,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
        ]);
    }
    SparseStructureSample {
        source: SparseStructureStageSource::Synthetic,
        sampler_config: sample_cfg,
        sigma_min,
        step_count: trace.steps,
        resolution,
        flow_resolution,
        flow_channels,
        noise,
        step_0_x_t: trace.step_0_x_t,
        step_mid_x_t: trace.step_mid_x_t,
        step_last_x_t: trace.step_last_x_t,
        latent,
        coords,
    }
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_sparse_structure_with_model(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    coords_override: Option<&[[u32; 4]]>,
    cond_override: Option<&[f32]>,
    neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    capture_sampler_trace: bool,
    sparse_flow: &SparseStructureFlowRuntime,
) -> Option<SparseStructureSample> {
    let config = sparse_flow.config();
    let flow_resolution = config.resolution;
    let channels = config.in_channels;
    let flow_voxels = flow_resolution * flow_resolution * flow_resolution;
    let noise = dense_noise_with_override(
        rng,
        channels * flow_voxels,
        noise_override,
        "sparse_runtime",
    );

    let cond_tokens = 32 * 32 + 5;
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "sparse_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: sparse flow cond override rejected ({err}); using synthetic sparse stage fallback."
            );
            return None;
        }
    };
    let neg_cond = match dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "sparse_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: sparse flow neg-cond override rejected ({err}); using synthetic sparse stage fallback."
            );
            return None;
        }
    };
    let cond_tensor = match sparse_flow.prepare_condition(cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: sparse flow cond preparation failed ({err}); using synthetic sparse stage fallback."
            );
            return None;
        }
    };
    let neg_cond_tensor = match sparse_flow.prepare_condition(neg_cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: sparse flow negative cond preparation failed ({err}); using synthetic sparse stage fallback."
            );
            return None;
        }
    };
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    let trace = match sparse_flow.sample_with_trace(
        noise.as_slice(),
        sample_cfg,
        sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        None,
        capture_sampler_trace,
    ) {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!(
                "burn_trellis: sparse flow model prediction failed ({err}); using synthetic sparse stage fallback."
            );
            return None;
        }
    };
    let latent = trace.samples;

    let occupancy = latent_to_occupancy(&latent, channels, flow_resolution);
    let upsampled = upsample_occupancy(occupancy.as_slice(), flow_resolution, resolution);
    let max_sparse_coords = runtime_max_sparse_coords();
    let mut coords = if let Some(override_coords) = coords_override {
        if runtime_stage_debug_enabled() {
            eprintln!(
                "burn_trellis: sparse runtime using hook coord override rows={}",
                override_coords.len()
            );
        }
        override_coords.to_vec()
    } else {
        let mut sampled =
            occupancy_to_coords(upsampled.as_slice(), resolution, 0.5, max_sparse_coords);
        if sampled.is_empty() {
            sampled.push([
                0,
                (resolution / 2) as u32,
                (resolution / 2) as u32,
                (resolution / 2) as u32,
            ]);
        }
        if let Some(limit) = max_sparse_coords {
            eprintln!(
                "burn_trellis: sparse coords after threshold/cap = {} (limit={})",
                sampled.len(),
                limit
            );
        }
        sampled
    };
    if coords.is_empty() {
        coords.push([
            0,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
        ]);
    }
    Some(SparseStructureSample {
        source: match sparse_flow.backend_name() {
            "wgpu" => SparseStructureStageSource::RuntimeModelWgpu,
            _ => SparseStructureStageSource::RuntimeModelCpu,
        },
        sampler_config: sample_cfg,
        sigma_min,
        step_count: trace.steps,
        resolution,
        flow_resolution,
        flow_channels: channels,
        noise,
        step_0_x_t: trace.step_0_x_t,
        step_mid_x_t: trace.step_mid_x_t,
        step_last_x_t: trace.step_last_x_t,
        latent,
        coords,
    })
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_shape_slat_with_model(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    capture_sampler_trace: bool,
    shape_flow: &SparseStructureFlowRuntime,
) -> Option<ShapeSLatSample> {
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    if coords.is_empty() {
        return Some(ShapeSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: capture_sampler_trace.then_some(Vec::new()),
            features: Vec::new(),
            noise: Vec::new(),
            step_0_x_t: Vec::new(),
            step_mid_x_t: Vec::new(),
            step_last_x_t: Vec::new(),
            coords: Vec::new(),
        });
    }
    let config = shape_flow.config();
    let dense_resolution = config.resolution.max(1);
    let voxel_count = dense_resolution * dense_resolution * dense_resolution;
    if voxel_count == 0 || config.out_channels == 0 {
        return None;
    }
    let feature_channels = 32usize.min(config.out_channels);
    let dense_indices = coords
        .iter()
        .map(|coord| map_coord_to_dense_flat(*coord, sparse_resolution, dense_resolution))
        .collect::<Vec<_>>();

    let noise = build_dense_runtime_noise(
        rng,
        config.out_channels,
        voxel_count,
        noise_dense_override,
        noise_override,
        coords,
        sparse_resolution,
        dense_resolution,
        "shape_slat_runtime",
    );

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "shape_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat cond override rejected ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };
    let neg_cond = match dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "shape_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat neg-cond override rejected ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };
    let cond_tensor = match shape_flow.prepare_condition(cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat cond preparation failed ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };
    let neg_cond_tensor = match shape_flow.prepare_condition(neg_cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat negative cond preparation failed ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };
    let trace = match shape_flow.sample_rows_with_trace(
        noise.as_slice(),
        sample_cfg,
        sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        None,
        dense_indices.as_slice(),
        feature_channels,
        capture_sampler_trace,
    ) {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat runtime prediction failed ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };

    let mut features = Vec::with_capacity(coords.len());
    let mut noise_rows = Vec::with_capacity(coords.len());
    let mut step_0_rows = Vec::with_capacity(coords.len());
    let mut step_mid_rows = Vec::with_capacity(coords.len());
    let mut step_last_rows = Vec::with_capacity(coords.len());
    let gathered_channels = feature_channels.min(trace.row_channels);
    for (row_idx, dense_idx) in dense_indices.iter().copied().enumerate() {
        let gathered_base = row_idx.saturating_mul(trace.row_channels);
        let mut row = [0.0f32; 32];
        let mut noise_row = [0.0f32; 32];
        let mut step_0_row = [0.0f32; 32];
        let mut step_mid_row = [0.0f32; 32];
        let mut step_last_row = [0.0f32; 32];
        for ch in 0..gathered_channels {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            let offset = ch * voxel_count + dense_idx;
            let sampled = trace.samples[gathered_base + ch];
            row[ch] = sampled * std + mean;
            noise_row[ch] = noise[offset];
            step_0_row[ch] = trace.step_0_x_t[gathered_base + ch];
            step_mid_row[ch] = trace.step_mid_x_t[gathered_base + ch];
            step_last_row[ch] = trace.step_last_x_t[gathered_base + ch];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
    }
    Some(ShapeSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution,
        dense_channels: config.out_channels,
        dense_noise: capture_sampler_trace.then_some(noise),
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        coords: coords.to_vec(),
    })
}

#[cfg(feature = "runtime-model")]
#[allow(clippy::too_many_arguments)]
fn sample_tex_slat_with_model(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    noise_dense_override: Option<&[f32]>,
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    capture_sampler_trace: bool,
    tex_flow: &SparseStructureFlowRuntime,
) -> Option<TexSLatSample> {
    let (_, sample_cfg, sigma_min) = resolve_sampler_settings(sampler_config, sampler_override);
    if shape_slat.coords.is_empty() {
        return Some(TexSLatSample {
            sampler_config: sample_cfg,
            sigma_min,
            step_count: sample_cfg.steps,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: capture_sampler_trace.then_some(Vec::new()),
            features: Vec::new(),
            noise: Vec::new(),
            step_0_x_t: Vec::new(),
            step_mid_x_t: Vec::new(),
            step_last_x_t: Vec::new(),
            shape_slat_cond: Vec::new(),
            coords: Vec::new(),
        });
    }
    let config = tex_flow.config();
    let dense_resolution = config.resolution.max(1);
    let voxel_count = dense_resolution * dense_resolution * dense_resolution;
    if voxel_count == 0 || config.out_channels == 0 {
        return None;
    }
    let feature_channels = 32usize.min(config.out_channels);
    let dense_indices = shape_slat
        .coords
        .iter()
        .map(|coord| map_coord_to_dense_flat(*coord, sparse_resolution, dense_resolution))
        .collect::<Vec<_>>();
    let concat_channels = config.in_channels.saturating_sub(config.out_channels);
    if concat_channels == 0 {
        eprintln!(
            "burn_trellis: tex flow runtime has no concat channels; using synthetic tex stage fallback."
        );
        return None;
    }

    let mut concat_dense = vec![0.0f32; concat_channels * voxel_count];
    let mut concat_counts = vec![0u32; voxel_count];
    for (idx, coord) in shape_slat.coords.iter().enumerate() {
        let dense_idx = map_coord_to_dense_flat(*coord, sparse_resolution, dense_resolution);
        concat_counts[dense_idx] = concat_counts[dense_idx].saturating_add(1);
        let shape_feat = shape_slat.features[idx];
        for ch in 0..concat_channels.min(32) {
            let mean = shape_normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = shape_normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            let normalized = (shape_feat[ch] - mean) / std;
            concat_dense[ch * voxel_count + dense_idx] += normalized;
        }
    }
    for voxel in 0..voxel_count {
        let count = concat_counts[voxel];
        if count == 0 {
            continue;
        }
        let inv = 1.0 / count as f32;
        for ch in 0..concat_channels {
            concat_dense[ch * voxel_count + voxel] *= inv;
        }
    }

    let noise = build_dense_runtime_noise(
        rng,
        config.out_channels,
        voxel_count,
        noise_dense_override,
        noise_override,
        shape_slat.coords.as_slice(),
        sparse_resolution,
        dense_resolution,
        "tex_slat_runtime",
    );

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = match dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "tex_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat cond override rejected ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };
    let neg_cond = match dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "tex_slat_runtime",
    ) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat neg-cond override rejected ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };
    let cond_tensor = match tex_flow.prepare_condition(cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat cond preparation failed ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };
    let neg_cond_tensor = match tex_flow.prepare_condition(neg_cond.as_slice(), cond_tokens) {
        Ok(cond) => cond,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat negative cond preparation failed ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };
    let trace = match tex_flow.sample_rows_with_trace(
        noise.as_slice(),
        sample_cfg,
        sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        Some(concat_dense.as_slice()),
        dense_indices.as_slice(),
        feature_channels,
        capture_sampler_trace,
    ) {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat runtime prediction failed ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };

    let mut features = Vec::with_capacity(shape_slat.coords.len());
    let mut noise_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_0_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_mid_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_last_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut shape_cond_rows = Vec::with_capacity(shape_slat.coords.len());
    let gathered_channels = feature_channels.min(trace.row_channels);
    for (idx, dense_idx) in dense_indices.iter().copied().enumerate() {
        let gathered_base = idx.saturating_mul(trace.row_channels);
        let mut row = [0.0f32; 32];
        let mut noise_row = [0.0f32; 32];
        let mut step_0_row = [0.0f32; 32];
        let mut step_mid_row = [0.0f32; 32];
        let mut step_last_row = [0.0f32; 32];
        let mut shape_cond = [0.0f32; 32];
        let shape_feat = shape_slat.features[idx];
        for ch in 0..32 {
            let shape_mean = shape_normalization.mean.get(ch).copied().unwrap_or(0.0);
            let shape_std = shape_normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            shape_cond[ch] = (shape_feat[ch] - shape_mean) / shape_std;
        }
        for ch in 0..gathered_channels {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            let offset = ch * voxel_count + dense_idx;
            let sampled = trace.samples[gathered_base + ch];
            row[ch] = sampled * std + mean;
            noise_row[ch] = noise[offset];
            step_0_row[ch] = trace.step_0_x_t[gathered_base + ch];
            step_mid_row[ch] = trace.step_mid_x_t[gathered_base + ch];
            step_last_row[ch] = trace.step_last_x_t[gathered_base + ch];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
        shape_cond_rows.push(shape_cond);
    }

    Some(TexSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution,
        dense_channels: config.out_channels,
        dense_noise: capture_sampler_trace.then_some(noise),
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        shape_slat_cond: shape_cond_rows,
        coords: shape_slat.coords.clone(),
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_shape_slat(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    _noise_dense_override: Option<&[f32]>,
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    capture_sampler_trace: bool,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] shape_flow: Option<&SparseStructureFlowRuntime>,
) -> Result<ShapeSLatSample, String> {
    #[cfg(feature = "runtime-model")]
    if let Some(shape_flow) = shape_flow
        && let Some(sample) = sample_shape_slat_with_model(
            preprocess,
            coords,
            rng,
            noise_override,
            _noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            shape_flow,
        )
    {
        return Ok(sample);
    }
    if parity_strict {
        return Err(
            "burn_trellis parity strict mode: shape_slat stage would use synthetic fallback"
                .to_string(),
        );
    }

    let mut features = Vec::with_capacity(coords.len());
    let mut noise_rows = Vec::with_capacity(coords.len());
    let mut step_0_rows = Vec::with_capacity(coords.len());
    let mut step_mid_rows = Vec::with_capacity(coords.len());
    let mut step_last_rows = Vec::with_capacity(coords.len());
    let (sampler, sample_cfg, sigma_min) =
        resolve_sampler_settings(sampler_config, sampler_override);
    let override_noise_map = noise_override.map(sparse_row_noise_map);
    for coord in coords {
        let base = sample_pixel_luma(preprocess, coord[1], coord[2], coord[3]);
        let noise = override_noise_map
            .as_ref()
            .and_then(|map| map.get(&pack_coord(coord[1], coord[2], coord[3])))
            .map(|row| row.to_vec())
            .unwrap_or_else(|| (0..32).map(|_| rng.next_normal_f32()).collect::<Vec<_>>());
        let target = [base; 32];
        let trace = sampler.sample_with_trace_mode(
            &noise,
            sample_cfg,
            capture_sampler_trace,
            |x_t, _t, cond| {
                let mut out = vec![0.0f32; x_t.len()];
                for idx in 0..out.len() {
                    let target_value = if cond { target[idx] } else { 0.0 };
                    out[idx] = x_t[idx] - target_value;
                }
                out
            },
        );
        let sampled = trace.samples;
        let mut row = [0.0f32; 32];
        let mut noise_row = [0.0f32; 32];
        let mut step_0_row = [0.0f32; 32];
        let mut step_mid_row = [0.0f32; 32];
        let mut step_last_row = [0.0f32; 32];
        for idx in 0..32 {
            let mean = normalization.mean.get(idx).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(idx)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            row[idx] = sampled[idx] * std + mean;
            noise_row[idx] = noise[idx];
            step_0_row[idx] = trace.step_0_x_t[idx];
            step_mid_row[idx] = trace.step_mid_x_t[idx];
            step_last_row[idx] = trace.step_last_x_t[idx];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
    }
    Ok(ShapeSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        coords: coords.to_vec(),
    })
}

#[allow(clippy::too_many_arguments)]
fn sample_tex_slat(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    _noise_dense_override: Option<&[f32]>,
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    sampler_override: Option<SamplerConfigOverride>,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    capture_sampler_trace: bool,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] tex_flow: Option<&SparseStructureFlowRuntime>,
) -> Result<TexSLatSample, String> {
    #[cfg(feature = "runtime-model")]
    if let Some(tex_flow) = tex_flow
        && let Some(sample) = sample_tex_slat_with_model(
            preprocess,
            shape_slat,
            rng,
            noise_override,
            _noise_dense_override,
            _cond_overrides,
            sampler_config,
            sampler_override,
            shape_normalization,
            normalization,
            _sparse_resolution,
            capture_sampler_trace,
            tex_flow,
        )
    {
        return Ok(sample);
    }
    if parity_strict {
        return Err(
            "burn_trellis parity strict mode: tex_slat stage would use synthetic fallback"
                .to_string(),
        );
    }

    let (sampler, sample_cfg, sigma_min) =
        resolve_sampler_settings(sampler_config, sampler_override);
    let mut features = Vec::with_capacity(shape_slat.coords.len());
    let mut noise_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_0_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_mid_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_last_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut shape_cond_rows = Vec::with_capacity(shape_slat.coords.len());
    let override_noise_map = noise_override.map(sparse_row_noise_map);
    for (idx, coord) in shape_slat.coords.iter().enumerate() {
        let luma = sample_pixel_luma(preprocess, coord[1], coord[2], coord[3]);
        let shape_hint = shape_slat.features[idx];
        let mut shape_cond = [0.0f32; 32];
        for ch in 0..32 {
            let mean = shape_normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = shape_normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            shape_cond[ch] = (shape_hint[ch] - mean) / std;
        }
        let noise = override_noise_map
            .as_ref()
            .and_then(|map| map.get(&pack_coord(coord[1], coord[2], coord[3])))
            .map(|row| row.to_vec())
            .unwrap_or_else(|| (0..32).map(|_| rng.next_normal_f32()).collect::<Vec<_>>());
        let target = (0..32)
            .map(|ch| 0.75 * luma + 0.25 * shape_cond[ch].tanh())
            .collect::<Vec<_>>();
        let trace = sampler.sample_with_trace_mode(
            &noise,
            sample_cfg,
            capture_sampler_trace,
            |x_t, _t, cond| {
                let mut out = vec![0.0f32; x_t.len()];
                for ch in 0..out.len() {
                    let target_value = if cond { target[ch] } else { 0.0 };
                    out[ch] = x_t[ch] - target_value;
                }
                out
            },
        );
        let sampled = trace.samples;
        let mut row = [0.0f32; 32];
        let mut noise_row = [0.0f32; 32];
        let mut step_0_row = [0.0f32; 32];
        let mut step_mid_row = [0.0f32; 32];
        let mut step_last_row = [0.0f32; 32];
        for ch in 0..32 {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            row[ch] = sampled[ch] * std + mean;
            noise_row[ch] = noise[ch];
            step_0_row[ch] = trace.step_0_x_t[ch];
            step_mid_row[ch] = trace.step_mid_x_t[ch];
            step_last_row[ch] = trace.step_last_x_t[ch];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
        shape_cond_rows.push(shape_cond);
    }
    Ok(TexSLatSample {
        sampler_config: sample_cfg,
        sigma_min,
        step_count: sample_cfg.steps,
        dense_resolution: 0,
        dense_channels: 0,
        dense_noise: None,
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        shape_slat_cond: shape_cond_rows,
        coords: shape_slat.coords.clone(),
    })
}

fn decode_latent_to_outputs(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] shape_decoder: Option<&FdgDecoderRuntime>,
    #[cfg(feature = "runtime-model")] tex_decoder: Option<&SparseUnetVaeDecoderRuntime>,
) -> Result<DecodedLatentOutput, String> {
    #[cfg(feature = "runtime-model")]
    {
        let Some(shape_decoder) = shape_decoder else {
            if parity_strict {
                return Err(
                    "burn_trellis: shape runtime decoder is required (missing `shape_slat_decoder` runtime)"
                        .to_string(),
                );
            }
            eprintln!(
                "burn_trellis: shape runtime decoder missing; using canonical-cube decode fallback"
            );
            return Ok(decoded_fallback_output());
        };
        let Some(tex_decoder) = tex_decoder else {
            if parity_strict {
                return Err(
                    "burn_trellis: tex runtime decoder is required (missing `tex_slat_decoder` runtime)"
                        .to_string(),
                );
            }
            eprintln!(
                "burn_trellis: tex runtime decoder missing; using canonical-cube decode fallback"
            );
            return Ok(decoded_fallback_output());
        };
        decode_latent_with_runtime_decoders(shape, tex, pipeline_type, parity_strict, shape_decoder, tex_decoder)
            .or_else(|err| {
                if parity_strict {
                    Err(format!("burn_trellis: runtime decode pipeline failed: {err}"))
                } else {
                    eprintln!("burn_trellis: runtime decode pipeline failed ({err}); using canonical-cube decode fallback");
                    Ok(decoded_fallback_output())
                }
            })
    }

    #[cfg(not(feature = "runtime-model"))]
    {
        let _ = (shape, tex, pipeline_type, parity_strict);
        Err("burn_trellis: TRELLIS decode requires `runtime-model` feature".to_string())
    }
}

fn decoded_fallback_output() -> DecodedLatentOutput {
    DecodedLatentOutput {
        mesh: canonical_cube(),
        shape_subs: Vec::new(),
        tex_voxels: DecodeTexVoxelSample {
            coords: Vec::new(),
            feats: Vec::new(),
            spatial_shape: [1, 1, 1],
        },
        pbr: None,
    }
}

#[cfg(feature = "runtime-model")]
fn decode_latent_with_runtime_decoders(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    parity_strict: bool,
    shape_decoder: &FdgDecoderRuntime,
    tex_decoder: &SparseUnetVaeDecoderRuntime,
) -> Result<DecodedLatentOutput, String> {
    let stage_debug = runtime_stage_debug_enabled();
    let count = shape
        .coords
        .len()
        .min(shape.features.len())
        .min(tex.features.len());
    if count == 0 {
        if parity_strict {
            return Err(
                "parity strict mode: runtime decode received empty shape/tex latent rows"
                    .to_string(),
            );
        }
        return Ok(decoded_fallback_output());
    }
    if shape_decoder.out_channels() < 7 || tex_decoder.out_channels() < 6 {
        return Err(format!(
            "decoder channel mismatch: shape_out={} tex_out={}",
            shape_decoder.out_channels(),
            tex_decoder.out_channels()
        ));
    }
    if stage_debug {
        eprintln!("burn_trellis: decode runtime begin (rows={count})");
    }
    let shape_rows = &shape.features[..count];
    let tex_rows = &tex.features[..count];
    let shape_decode_start = Instant::now();
    let shape_decoded = shape_decoder
        .decode_sparse(&shape.coords[..count], shape_rows)
        .map_err(|err| format!("shape runtime decoder failed: {err}"))?;
    if stage_debug {
        eprintln!(
            "burn_trellis: decode runtime shape-decoder complete ({:.2} ms, subs={}, coords={})",
            shape_decode_start.elapsed().as_secs_f64() * 1000.0,
            shape_decoded.subdivisions.len(),
            shape_decoded.coords.len()
        );
    }

    let tex_decode_start = Instant::now();
    let tex_decoded = tex_decoder
        .decode_with_guidance(
            &tex.coords[..count],
            tex_rows,
            shape_decoded.subdivisions.as_slice(),
        )
        .map_err(|err| format!("tex runtime decoder failed: {err}"))?;
    if stage_debug {
        eprintln!(
            "burn_trellis: decode runtime tex-decoder complete ({:.2} ms, coords={})",
            tex_decode_start.elapsed().as_secs_f64() * 1000.0,
            tex_decoded.coords.len()
        );
    }

    let final_resolution = final_resolution_for_pipeline(pipeline_type);
    let coords = shape_decoded.coords;
    let attr_merge_start = Instant::now();
    let mut tex_by_coord = HashMap::with_capacity(tex_decoded.coords.len() * 2);
    for (coord, attr) in tex_decoded
        .coords
        .iter()
        .copied()
        .zip(tex_decoded.attrs.iter().copied())
    {
        tex_by_coord.insert(coord, attr);
    }
    let voxel_attrs = coords
        .iter()
        .map(|coord| tex_by_coord.get(coord).copied().unwrap_or([0.5; 6]))
        .collect::<Vec<_>>();
    if stage_debug {
        eprintln!(
            "burn_trellis: decode runtime attr merge complete ({:.2} ms)",
            attr_merge_start.elapsed().as_secs_f64() * 1000.0
        );
    }

    let grid_size = [
        final_resolution as u32,
        final_resolution as u32,
        final_resolution as u32,
    ];
    let mesh_start = Instant::now();
    let (vertices, faces) = flexible_dual_grid_to_mesh(
        &coords,
        shape_decoded.vertices.as_slice(),
        shape_decoded.intersected.as_slice(),
        Some(shape_decoded.quad_lerp.as_slice()),
        grid_size,
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
    );
    if stage_debug {
        eprintln!(
            "burn_trellis: decode runtime mesh complete ({:.2} ms, vertices={}, faces={})",
            mesh_start.elapsed().as_secs_f64() * 1000.0,
            vertices.len(),
            faces.len()
        );
    }
    let (uvs, pbr_textures, pbr_debug) = bake_pbr_from_voxels(
        vertices.as_slice(),
        faces.as_slice(),
        coords.as_slice(),
        voxel_attrs.as_slice(),
        final_resolution as u32,
    );
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mesh = if vertices.is_empty() || faces.is_empty() {
        if parity_strict {
            return Err("parity strict mode: runtime decode produced empty mesh".to_string());
        }
        canonical_cube()
    } else {
        Mesh {
            vertices,
            faces,
            uvs,
            material,
            pbr_textures,
        }
    };

    let shape_subs = shape_decoded
        .subdivisions
        .iter()
        .map(runtime_subdivision_to_sample)
        .collect::<Vec<_>>();
    let tex_spatial = spatial_shape_from_sparse_coords(coords.as_slice());

    Ok(DecodedLatentOutput {
        mesh,
        shape_subs,
        tex_voxels: DecodeTexVoxelSample {
            coords,
            feats: voxel_attrs,
            spatial_shape: tex_spatial,
        },
        pbr: Some(pbr_debug),
    })
}

struct Lcg {
    state: u64,
    cached_normal: Option<f32>,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32 + 0.5) * (1.0 / 4_294_967_296.0)
    }

    fn next_open01(&mut self) -> f32 {
        self.next_f32().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON)
    }

    fn next_normal_f32(&mut self) -> f32 {
        if let Some(cached) = self.cached_normal.take() {
            return cached;
        }
        let u1 = self.next_open01();
        let u2 = self.next_f32();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.cached_normal = Some(z1);
        z0
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "runtime-model")]
    use std::collections::HashMap;
    #[cfg(feature = "runtime-model")]
    use std::path::PathBuf;

    use super::{
        FlowEulerSampleConfig, ShapeSLatSample, TexSLatSample, bake_pbr_from_voxels,
        decode_latent_to_outputs, summarize_material,
    };
    #[cfg(feature = "runtime-model")]
    use crate::hook_diff::{HookSnapshot, compute_stats};
    use crate::mesh::MeshPbrTextures;
    #[cfg(feature = "runtime-model")]
    use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
    #[cfg(feature = "runtime-model")]
    use crate::runtime_model::fdg_decoder::FdgDecoderRuntime;
    #[cfg(feature = "runtime-model")]
    use crate::runtime_model::sparse_decoder::SparseSubdivisionLogits;
    #[cfg(feature = "runtime-model")]
    use crate::runtime_model::sparse_unet_vae_decoder::SparseUnetVaeDecoderRuntime;
    #[cfg(feature = "runtime-model")]
    use crate::trellis_config::TrellisPipelineConfig;

    #[cfg(feature = "runtime-model")]
    fn env_flag(name: &str) -> bool {
        std::env::var(name)
            .ok()
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    }

    fn env_usize(name: &str) -> Option<usize> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
    }

    fn env_f32(name: &str) -> Option<f32> {
        std::env::var(name)
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
    }

    fn dummy_textures() -> MeshPbrTextures {
        let rgba = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        MeshPbrTextures {
            base_color: crate::mesh::MeshTexture {
                width: 2,
                height: 2,
                rgba8: rgba.clone(),
            },
            metallic_roughness: crate::mesh::MeshTexture {
                width: 2,
                height: 2,
                rgba8: vec![
                    0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255, 0, 220, 20, 255,
                ],
            },
            normal: None,
            emissive: None,
            occlusion: None,
        }
    }

    #[test]
    fn pbr_bake_produces_textures_and_uvs() {
        let vertices = vec![[-0.5, 0.0, -0.5], [0.5, 0.0, -0.5], [0.0, 0.0, 0.5]];
        let faces = vec![[0, 1, 2]];
        let vox_coords = vec![[0, 16, 16, 16], [0, 20, 16, 16], [0, 16, 20, 16]];
        let vox_attrs = vec![
            [0.8, 0.2, 0.1, 0.1, 0.8, 1.0],
            [0.1, 0.8, 0.2, 0.3, 0.6, 1.0],
            [0.2, 0.1, 0.8, 0.5, 0.4, 1.0],
        ];

        let (uvs, textures, debug) =
            bake_pbr_from_voxels(&vertices, &faces, &vox_coords, &vox_attrs, 32);
        assert_eq!(uvs.len(), vertices.len());
        let textures = textures.expect("pbr textures should exist");
        assert!(textures.base_color.width >= 64);
        assert_eq!(
            textures.base_color.rgba8.len(),
            (textures.base_color.width * textures.base_color.height * 4) as usize
        );
        assert!(debug.raster_mask.iter().any(|value| *value != 0));
    }

    #[test]
    fn pbr_quantization_tracks_float_buffers() {
        let vertices = vec![
            [-0.5, 0.0, -0.5],
            [0.5, 0.0, -0.5],
            [0.5, 0.0, 0.5],
            [-0.5, 0.0, 0.5],
        ];
        let faces = vec![[0, 1, 2], [0, 2, 3]];
        let vox_coords = vec![[0, 8, 8, 8], [0, 16, 8, 8], [0, 16, 16, 8], [0, 8, 16, 8]];
        let vox_attrs = vec![
            [0.2, 0.3, 0.4, 0.2, 0.7, 1.0],
            [0.5, 0.6, 0.7, 0.4, 0.5, 1.0],
            [0.8, 0.6, 0.3, 0.6, 0.4, 1.0],
            [0.4, 0.2, 0.1, 0.1, 0.8, 1.0],
        ];

        let (_, _, debug) = bake_pbr_from_voxels(&vertices, &faces, &vox_coords, &vox_attrs, 32);
        assert!(!debug.base_color_float.is_empty());
        assert_eq!(
            debug.base_color_float.len(),
            debug.texture_width * debug.texture_height
        );
        assert_eq!(
            debug.metallic_float.len(),
            debug.texture_width * debug.texture_height
        );

        for (idx, rgba) in debug.base_color_float.iter().enumerate() {
            let off = idx * 4;
            let expected = [
                (rgba[0].clamp(0.0, 1.0) * 255.0).round() as i32,
                (rgba[1].clamp(0.0, 1.0) * 255.0).round() as i32,
                (rgba[2].clamp(0.0, 1.0) * 255.0).round() as i32,
                (debug.alpha_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32,
            ];
            for (channel, expected_value) in expected.iter().enumerate() {
                let actual = debug.base_color_rgba_u8[off + channel] as i32;
                assert!(
                    (actual - *expected_value).abs() <= 1,
                    "base channel mismatch idx={idx} ch={channel}: actual={actual}, expected={}",
                    expected_value
                );
            }

            let expected_metallic =
                (debug.metallic_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32;
            let expected_roughness =
                (debug.roughness_float[idx].clamp(0.0, 1.0) * 255.0).round() as i32;
            let mr = &debug.metallic_roughness_u8[off..off + 4];
            assert!((mr[1] as i32 - expected_roughness).abs() <= 1);
            assert!((mr[2] as i32 - expected_metallic).abs() <= 1);
        }
    }

    #[test]
    fn material_summary_prefers_texture_data_when_available() {
        let textures = dummy_textures();
        let material = summarize_material(&[[0.0; 6]], Some(&textures)).expect("material");
        assert!(material.base_color[0] > 0.1);
        assert!(material.alpha > 0.8);
    }

    #[cfg(feature = "runtime-model")]
    #[test]
    fn decode_missing_runtime_decoders_falls_back_when_not_strict() {
        let shape = ShapeSLatSample {
            sampler_config: FlowEulerSampleConfig {
                steps: 1,
                rescale_t: 1.0,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.0, 1.0],
            },
            sigma_min: 1.0e-3,
            step_count: 1,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: None,
            features: vec![[0.0; 32]],
            noise: vec![[0.0; 32]],
            step_0_x_t: vec![[0.0; 32]],
            step_mid_x_t: vec![[0.0; 32]],
            step_last_x_t: vec![[0.0; 32]],
            coords: vec![[0, 0, 0, 0]],
        };
        let tex = TexSLatSample {
            sampler_config: FlowEulerSampleConfig {
                steps: 1,
                rescale_t: 1.0,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.0, 1.0],
            },
            sigma_min: 1.0e-3,
            step_count: 1,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: None,
            features: vec![[0.0; 32]],
            noise: vec![[0.0; 32]],
            step_0_x_t: vec![[0.0; 32]],
            step_mid_x_t: vec![[0.0; 32]],
            step_last_x_t: vec![[0.0; 32]],
            shape_slat_cond: vec![[0.0; 32]],
            coords: vec![[0, 0, 0, 0]],
        };
        let decoded = decode_latent_to_outputs(&shape, &tex, "512", false, None, None)
            .expect("non-strict decode should use fallback output when decoders are missing");
        assert!(!decoded.mesh.vertices.is_empty());
        assert!(!decoded.mesh.faces.is_empty());
    }

    #[cfg(feature = "runtime-model")]
    #[test]
    fn decode_missing_runtime_decoders_errors_when_strict() {
        let shape = ShapeSLatSample {
            sampler_config: FlowEulerSampleConfig {
                steps: 1,
                rescale_t: 1.0,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.0, 1.0],
            },
            sigma_min: 1.0e-3,
            step_count: 1,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: None,
            features: vec![[0.0; 32]],
            noise: vec![[0.0; 32]],
            step_0_x_t: vec![[0.0; 32]],
            step_mid_x_t: vec![[0.0; 32]],
            step_last_x_t: vec![[0.0; 32]],
            coords: vec![[0, 0, 0, 0]],
        };
        let tex = TexSLatSample {
            sampler_config: FlowEulerSampleConfig {
                steps: 1,
                rescale_t: 1.0,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.0, 1.0],
            },
            sigma_min: 1.0e-3,
            step_count: 1,
            dense_resolution: 0,
            dense_channels: 0,
            dense_noise: None,
            features: vec![[0.0; 32]],
            noise: vec![[0.0; 32]],
            step_0_x_t: vec![[0.0; 32]],
            step_mid_x_t: vec![[0.0; 32]],
            step_last_x_t: vec![[0.0; 32]],
            shape_slat_cond: vec![[0.0; 32]],
            coords: vec![[0, 0, 0, 0]],
        };
        let err = decode_latent_to_outputs(&shape, &tex, "512", true, None, None)
            .expect_err("strict decode should fail when runtime decoders are missing");
        assert!(err.contains("shape runtime decoder is required"));
    }

    #[cfg(feature = "runtime-model")]
    #[test]
    fn runtime_decoder_hook_alignment_report() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let reference_path = std::env::var("TRELLIS2_DECODER_REFERENCE_HOOK")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                root.join("assets/hooks/trellis2_full_reference_alpha_512.safetensors")
            });
        if !reference_path.exists() {
            eprintln!(
                "Skipping runtime_decoder_hook_alignment_report: missing reference hook '{}'",
                reference_path.display()
            );
            return;
        }
        let reference =
            HookSnapshot::from_file(&reference_path).expect("reference hook should load");

        let has_decode_inputs = reference
            .tensors
            .contains_key("decode_shape_slat.input.coords")
            && reference
                .tensors
                .contains_key("decode_shape_slat.input.feats");
        let (shape_coords, shape_feats) = if has_decode_inputs {
            let coords = tensor_to_coords4(
                reference
                    .tensors
                    .get("decode_shape_slat.input.coords")
                    .expect("missing decode_shape_slat.input.coords"),
            )
            .expect("decode input coords should decode");
            let feats = tensor_to_rows::<32>(
                reference
                    .tensors
                    .get("decode_shape_slat.input.feats")
                    .expect("missing decode_shape_slat.input.feats"),
            )
            .expect("decode input feats should decode");
            (coords, feats)
        } else {
            eprintln!(
                "runtime_decoder_hook_alignment_report: reference hook missing decode_shape_slat.input.*; using sample_shape_slat.slat.* fallback (subdivision logit comparisons may be context-mismatched)."
            );
            let coords = tensor_to_coords4(
                reference
                    .tensors
                    .get("sample_shape_slat.slat.coords")
                    .expect("missing sample_shape_slat.slat.coords"),
            )
            .expect("shape coords should decode");
            let feats = tensor_to_rows::<32>(
                reference
                    .tensors
                    .get("sample_shape_slat.slat.feats")
                    .expect("missing sample_shape_slat.slat.feats"),
            )
            .expect("shape feats should decode");
            (coords, feats)
        };
        let tex_coords = tensor_to_coords4(
            reference
                .tensors
                .get("sample_tex_slat.slat.coords")
                .expect("missing sample_tex_slat.slat.coords"),
        )
        .expect("tex coords should decode");
        let tex_feats = tensor_to_rows::<32>(
            reference
                .tensors
                .get("sample_tex_slat.slat.feats")
                .expect("missing sample_tex_slat.slat.feats"),
        )
        .expect("tex feats should decode");
        let reference_voxel_coords = tensor_to_coords4(
            reference
                .tensors
                .get("decode_tex_slat.voxels.coords")
                .expect("missing decode_tex_slat.voxels.coords"),
        )
        .expect("reference voxel coords should decode");
        let reference_voxel_feats = tensor_to_rows::<6>(
            reference
                .tensors
                .get("decode_tex_slat.voxels.feats")
                .expect("missing decode_tex_slat.voxels.feats"),
        )
        .expect("reference voxel feats should decode");
        let reference_subdivisions = load_reference_subdivisions(&reference)
            .expect("reference shape subdivisions should decode");

        let mut rows = shape_coords
            .len()
            .min(shape_feats.len())
            .min(tex_coords.len())
            .min(tex_feats.len());
        assert!(rows > 0, "reference hooks must contain slat rows");
        if let Ok(value) = std::env::var("TRELLIS2_DECODER_TEST_MAX_ROWS")
            && let Ok(cap) = value.trim().parse::<usize>()
            && cap > 0
            && rows > cap
        {
            rows = cap;
        }

        let weights_root = resolve_trellis2_weights_root(None);
        if !weights_root.exists() {
            eprintln!(
                "Skipping runtime_decoder_hook_alignment_report: missing weights root '{}'",
                weights_root.display()
            );
            return;
        }
        let image_large_root = resolve_trellis2_image_large_root(None);
        let image_large_root_opt = if image_large_root.exists() {
            Some(image_large_root)
        } else {
            None
        };

        let pipeline_bytes =
            std::fs::read(weights_root.join("pipeline.json")).expect("pipeline.json should load");
        let pipeline = TrellisPipelineConfig::from_json_bytes(pipeline_bytes.as_slice())
            .expect("pipeline config should parse");
        let shape_stem = pipeline
            .args
            .models
            .get("shape_slat_decoder")
            .expect("shape_slat_decoder model stem missing");
        let tex_stem = pipeline
            .args
            .models
            .get("tex_slat_decoder")
            .expect("tex_slat_decoder model stem missing");

        let shape_decoder = FdgDecoderRuntime::load_from_stem(
            weights_root.as_path(),
            image_large_root_opt.as_deref(),
            shape_stem.as_str(),
            false,
        )
        .expect("shape decoder should load");
        let tex_decoder = SparseUnetVaeDecoderRuntime::load_from_stem(
            weights_root.as_path(),
            image_large_root_opt.as_deref(),
            tex_stem.as_str(),
            false,
        )
        .expect("tex decoder should load");

        let shape_decoded = shape_decoder
            .decode_sparse(&shape_coords[..rows], &shape_feats[..rows])
            .expect("shape decoder should run");
        if has_decode_inputs {
            for (level, actual_sub) in shape_decoded.subdivisions.iter().enumerate() {
                if let Some(reference_sub) = reference_subdivisions.get(level) {
                    let (sub_stats, sub_overlap, actual_sub_rows, reference_sub_rows) =
                        compare_subdivision_overlap(actual_sub, reference_sub);
                    let (actual_min, actual_max, actual_mean) =
                        tensor_stats(actual_sub.logits.as_slice());
                    let (reference_min, reference_max, reference_mean) =
                        tensor_stats(reference_sub.logits.as_slice());
                    println!(
                        "runtime_decoder_hook_alignment_report shape_subdiv.level={} overlap={} actual_rows={} reference_rows={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e} actual[min,max,mean]=[{:.6e},{:.6e},{:.6e}] reference[min,max,mean]=[{:.6e},{:.6e},{:.6e}]",
                        level,
                        sub_overlap,
                        actual_sub_rows,
                        reference_sub_rows,
                        sub_stats.mean_abs,
                        sub_stats.max_abs,
                        sub_stats.rmse,
                        actual_min,
                        actual_max,
                        actual_mean,
                        reference_min,
                        reference_max,
                        reference_mean
                    );
                    if let Some(top_k) = env_usize("TRELLIS2_DECODER_SUBDIV_TOPK")
                        && top_k > 0
                    {
                        for (rank, entry) in top_subdivision_diffs(actual_sub, reference_sub, top_k)
                            .into_iter()
                            .enumerate()
                        {
                            println!(
                                "runtime_decoder_hook_alignment_report shape_subdiv.level={} top_diff.rank={} coord=[{},{},{},{}] child={} abs_diff={:.6e} actual={:.6e} reference={:.6e}",
                                level,
                                rank + 1,
                                entry.coord[0],
                                entry.coord[1],
                                entry.coord[2],
                                entry.coord[3],
                                entry.child,
                                entry.abs_diff,
                                entry.actual,
                                entry.reference
                            );
                        }
                    }
                }
            }
        }
        let tex_decoded = tex_decoder
            .decode_with_guidance(
                &tex_coords[..rows],
                &tex_feats[..rows],
                shape_decoded.subdivisions.as_slice(),
            )
            .expect("tex decoder should run");
        if env_flag("TRELLIS2_DECODER_DEBUG_REFERENCE_GUIDE")
            && shape_decoded.subdivisions.len() <= reference_subdivisions.len()
            && let Ok(tex_decoded_reference_guides) = tex_decoder.decode_with_guidance(
                &tex_coords[..rows],
                &tex_feats[..rows],
                &reference_subdivisions[..shape_decoded.subdivisions.len()],
            )
        {
            let (
                ref_guide_stats,
                ref_guide_overlap,
                ref_guide_actual_total,
                ref_guide_reference_total,
                _,
            ) = compare_tex_voxel_overlap(
                tex_decoded_reference_guides.coords.as_slice(),
                tex_decoded_reference_guides.attrs.as_slice(),
                reference_voxel_coords.as_slice(),
                reference_voxel_feats.as_slice(),
            );
            println!(
                "runtime_decoder_hook_alignment_report reference_guide overlap={} actual_voxels={} reference_voxels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
                ref_guide_overlap,
                ref_guide_actual_total,
                ref_guide_reference_total,
                ref_guide_stats.mean_abs,
                ref_guide_stats.max_abs,
                ref_guide_stats.rmse
            );
        }

        assert!(
            !shape_decoded.coords.is_empty(),
            "decoded shape coords should not be empty"
        );
        assert!(
            !tex_decoded.coords.is_empty(),
            "decoded tex coords should not be empty"
        );

        let (stats, overlap, actual_total, reference_total, per_channel) =
            compare_tex_voxel_overlap(
                tex_decoded.coords.as_slice(),
                tex_decoded.attrs.as_slice(),
                reference_voxel_coords.as_slice(),
                reference_voxel_feats.as_slice(),
            );
        println!(
            "runtime_decoder_hook_alignment_report overlap={} actual_voxels={} reference_voxels={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            overlap, actual_total, reference_total, stats.mean_abs, stats.max_abs, stats.rmse
        );
        for (channel, channel_stats) in per_channel.iter().enumerate() {
            println!(
                "runtime_decoder_hook_alignment_report channel={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
                channel, channel_stats.mean_abs, channel_stats.max_abs, channel_stats.rmse
            );
        }
        assert!(
            overlap > 0,
            "expected overlapping decode voxels with reference hooks"
        );
        assert!(
            stats.mean_abs.is_finite() && stats.max_abs.is_finite() && stats.rmse.is_finite(),
            "decoder diff stats must be finite"
        );
        if let Some(min_overlap) = env_usize("TRELLIS2_DECODER_MIN_OVERLAP") {
            assert!(
                overlap >= min_overlap,
                "decoder overlap {} below TRELLIS2_DECODER_MIN_OVERLAP={}",
                overlap,
                min_overlap
            );
        }
        if let Some(max_mean_abs) = env_f32("TRELLIS2_DECODER_MAX_MEAN_ABS") {
            assert!(
                stats.mean_abs <= max_mean_abs,
                "decoder mean_abs {:.6e} exceeded TRELLIS2_DECODER_MAX_MEAN_ABS={:.6e}",
                stats.mean_abs,
                max_mean_abs
            );
        }
        if let Some(max_rmse) = env_f32("TRELLIS2_DECODER_MAX_RMSE") {
            assert!(
                stats.rmse <= max_rmse,
                "decoder rmse {:.6e} exceeded TRELLIS2_DECODER_MAX_RMSE={:.6e}",
                stats.rmse,
                max_rmse
            );
        }
        if let Some(max_abs) = env_f32("TRELLIS2_DECODER_MAX_ABS") {
            assert!(
                stats.max_abs <= max_abs,
                "decoder max_abs {:.6e} exceeded TRELLIS2_DECODER_MAX_ABS={:.6e}",
                stats.max_abs,
                max_abs
            );
        }
    }

    #[cfg(feature = "runtime-model")]
    #[test]
    fn runtime_decoder_stage0_subdivision_alignment_report() {
        let reference_path = match std::env::var("TRELLIS2_DECODER_REFERENCE_HOOK") {
            Ok(path) => PathBuf::from(path),
            Err(_) => {
                eprintln!(
                    "Skipping runtime_decoder_stage0_subdivision_alignment_report: set TRELLIS2_DECODER_REFERENCE_HOOK to a stage0 alignment hook."
                );
                return;
            }
        };
        if !reference_path.exists() {
            eprintln!(
                "Skipping runtime_decoder_stage0_subdivision_alignment_report: missing reference hook '{}'",
                reference_path.display()
            );
            return;
        }
        let reference =
            HookSnapshot::from_file(&reference_path).expect("reference hook should load");

        let shape_coords = tensor_to_coords4(
            reference
                .tensors
                .get("sample_shape_slat.slat.coords")
                .expect("missing sample_shape_slat.slat.coords"),
        )
        .expect("shape coords should decode");
        let shape_feats = tensor_to_rows::<32>(
            reference
                .tensors
                .get("sample_shape_slat.slat.feats")
                .expect("missing sample_shape_slat.slat.feats"),
        )
        .expect("shape feats should decode");
        let reference_subdivisions = load_reference_subdivisions(&reference)
            .expect("reference shape subdivisions should decode");
        let Some(reference_stage0) = reference_subdivisions.first() else {
            eprintln!(
                "Skipping runtime_decoder_stage0_subdivision_alignment_report: no decode_shape_slat.subs.0 in '{}'",
                reference_path.display()
            );
            return;
        };

        let weights_root = resolve_trellis2_weights_root(None);
        if !weights_root.exists() {
            eprintln!(
                "Skipping runtime_decoder_stage0_subdivision_alignment_report: missing weights root '{}'",
                weights_root.display()
            );
            return;
        }
        let image_large_root = resolve_trellis2_image_large_root(None);
        let image_large_root_opt = if image_large_root.exists() {
            Some(image_large_root)
        } else {
            None
        };

        let pipeline_bytes =
            std::fs::read(weights_root.join("pipeline.json")).expect("pipeline.json should load");
        let pipeline = TrellisPipelineConfig::from_json_bytes(pipeline_bytes.as_slice())
            .expect("pipeline config should parse");
        let shape_stem = pipeline
            .args
            .models
            .get("shape_slat_decoder")
            .expect("shape_slat_decoder model stem missing");
        let shape_decoder = FdgDecoderRuntime::load_from_stem(
            weights_root.as_path(),
            image_large_root_opt.as_deref(),
            shape_stem.as_str(),
            false,
        )
        .expect("shape decoder should load");

        let stage0 = shape_decoder
            .stage0_subdivision_logits(shape_coords.as_slice(), shape_feats.as_slice())
            .expect("shape stage0 subdivision should run");
        let (stats, overlap, actual_rows, reference_rows) =
            compare_subdivision_overlap(&stage0, reference_stage0);
        println!(
            "runtime_decoder_stage0_subdivision_alignment_report overlap={} actual_rows={} reference_rows={} mean_abs={:.6e} max_abs={:.6e} rmse={:.6e}",
            overlap, actual_rows, reference_rows, stats.mean_abs, stats.max_abs, stats.rmse
        );
        assert!(
            overlap > 0,
            "expected overlapping stage0 subdivision coords"
        );
        assert!(
            stats.mean_abs.is_finite() && stats.max_abs.is_finite() && stats.rmse.is_finite(),
            "stage0 subdivision diff stats must be finite"
        );
    }

    #[cfg(feature = "runtime-model")]
    fn tensor_to_coords4(tensor: &crate::hook_diff::HookTensor) -> Result<Vec<[u32; 4]>, String> {
        if tensor.shape.len() != 2 || tensor.shape[1] != 4 {
            return Err(format!(
                "expected coords tensor shape [N,4], got {:?}",
                tensor.shape
            ));
        }
        let rows = tensor.shape[0];
        if tensor.data.len() != rows * 4 {
            return Err(format!(
                "coords tensor element count mismatch: expected {}, got {}",
                rows * 4,
                tensor.data.len()
            ));
        }
        let mut out = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let base = row_idx * 4;
            out.push([
                tensor.data[base].round().max(0.0) as u32,
                tensor.data[base + 1].round().max(0.0) as u32,
                tensor.data[base + 2].round().max(0.0) as u32,
                tensor.data[base + 3].round().max(0.0) as u32,
            ]);
        }
        Ok(out)
    }

    #[cfg(feature = "runtime-model")]
    fn tensor_to_rows<const C: usize>(
        tensor: &crate::hook_diff::HookTensor,
    ) -> Result<Vec<[f32; C]>, String> {
        if tensor.shape.len() != 2 || tensor.shape[1] != C {
            return Err(format!(
                "expected row tensor shape [N,{C}], got {:?}",
                tensor.shape
            ));
        }
        let rows = tensor.shape[0];
        if tensor.data.len() != rows * C {
            return Err(format!(
                "row tensor element count mismatch: expected {}, got {}",
                rows * C,
                tensor.data.len()
            ));
        }
        let mut out = Vec::with_capacity(rows);
        for row_idx in 0..rows {
            let base = row_idx * C;
            let mut row = [0.0f32; C];
            row.copy_from_slice(&tensor.data[base..base + C]);
            out.push(row);
        }
        Ok(out)
    }

    #[cfg(feature = "runtime-model")]
    fn tensor_to_spatial_shape3(tensor: &crate::hook_diff::HookTensor) -> Result<[u32; 3], String> {
        if tensor.shape.len() != 1 || tensor.shape[0] != 3 {
            return Err(format!(
                "expected spatial shape tensor [3], got {:?}",
                tensor.shape
            ));
        }
        if tensor.data.len() != 3 {
            return Err(format!(
                "spatial shape tensor element count mismatch: expected 3, got {}",
                tensor.data.len()
            ));
        }
        Ok([
            tensor.data[0].round().max(0.0) as u32,
            tensor.data[1].round().max(0.0) as u32,
            tensor.data[2].round().max(0.0) as u32,
        ])
    }

    #[cfg(feature = "runtime-model")]
    fn load_reference_subdivisions(
        hook: &HookSnapshot,
    ) -> Result<Vec<SparseSubdivisionLogits>, String> {
        let mut levels = Vec::new();
        for level in 0usize..16 {
            let coords_key = format!("decode_shape_slat.subs.{level}.coords");
            let feats_key = format!("decode_shape_slat.subs.{level}.feats");
            let spatial_key = format!("decode_shape_slat.subs.{level}.spatial_shape");
            let (Some(coords_tensor), Some(feats_tensor), Some(spatial_tensor)) = (
                hook.tensors.get(coords_key.as_str()),
                hook.tensors.get(feats_key.as_str()),
                hook.tensors.get(spatial_key.as_str()),
            ) else {
                break;
            };
            let coords = tensor_to_coords4(coords_tensor)?;
            let feats = tensor_to_rows::<8>(feats_tensor)?;
            let spatial_shape = tensor_to_spatial_shape3(spatial_tensor)?;
            if coords.len() != feats.len() {
                return Err(format!(
                    "reference subdivision level {} coords/feats mismatch: {} vs {}",
                    level,
                    coords.len(),
                    feats.len()
                ));
            }
            let mut logits = Vec::with_capacity(feats.len() * 8);
            for row in feats {
                logits.extend_from_slice(row.as_slice());
            }
            levels.push(SparseSubdivisionLogits {
                coords,
                logits,
                spatial_shape,
            });
        }
        Ok(levels)
    }

    #[cfg(feature = "runtime-model")]
    fn compare_subdivision_overlap(
        actual: &SparseSubdivisionLogits,
        reference: &SparseSubdivisionLogits,
    ) -> (crate::hook_diff::MetricStats, usize, usize, usize) {
        let mut actual_map: HashMap<[u32; 4], Vec<f32>> =
            HashMap::with_capacity(actual.coords.len().saturating_mul(2));
        for (idx, coord) in actual.coords.iter().copied().enumerate() {
            let row = &actual.logits[idx * 8..(idx + 1) * 8];
            actual_map.insert(coord, row.to_vec());
        }
        let mut reference_map: HashMap<[u32; 4], Vec<f32>> =
            HashMap::with_capacity(reference.coords.len().saturating_mul(2));
        for (idx, coord) in reference.coords.iter().copied().enumerate() {
            let row = &reference.logits[idx * 8..(idx + 1) * 8];
            reference_map.insert(coord, row.to_vec());
        }
        let mut actual_flat = Vec::new();
        let mut reference_flat = Vec::new();
        for (coord, reference_row) in &reference_map {
            if let Some(actual_row) = actual_map.get(coord) {
                actual_flat.extend_from_slice(actual_row.as_slice());
                reference_flat.extend_from_slice(reference_row.as_slice());
            }
        }
        let overlap = actual_flat.len() / 8;
        let stats = compute_stats(actual_flat.as_slice(), reference_flat.as_slice());
        (stats, overlap, actual_map.len(), reference_map.len())
    }

    #[cfg(feature = "runtime-model")]
    #[derive(Clone, Copy, Debug)]
    struct SubdivisionDiffEntry {
        coord: [u32; 4],
        child: usize,
        abs_diff: f32,
        actual: f32,
        reference: f32,
    }

    #[cfg(feature = "runtime-model")]
    fn top_subdivision_diffs(
        actual: &SparseSubdivisionLogits,
        reference: &SparseSubdivisionLogits,
        k: usize,
    ) -> Vec<SubdivisionDiffEntry> {
        if k == 0 {
            return Vec::new();
        }
        let mut actual_map: HashMap<[u32; 4], [f32; 8]> =
            HashMap::with_capacity(actual.coords.len().saturating_mul(2));
        for (idx, coord) in actual.coords.iter().copied().enumerate() {
            let mut row = [0.0f32; 8];
            row.copy_from_slice(&actual.logits[idx * 8..(idx + 1) * 8]);
            actual_map.insert(coord, row);
        }

        let mut out = Vec::new();
        for (idx, coord) in reference.coords.iter().copied().enumerate() {
            let Some(actual_row) = actual_map.get(&coord) else {
                continue;
            };
            let reference_row = &reference.logits[idx * 8..(idx + 1) * 8];
            for child in 0..8 {
                let actual_value = actual_row[child];
                let reference_value = reference_row[child];
                out.push(SubdivisionDiffEntry {
                    coord,
                    child,
                    abs_diff: (actual_value - reference_value).abs(),
                    actual: actual_value,
                    reference: reference_value,
                });
            }
        }
        out.sort_by(|a, b| b.abs_diff.total_cmp(&a.abs_diff));
        out.truncate(k);
        out
    }

    #[cfg(feature = "runtime-model")]
    fn tensor_stats(values: &[f32]) -> (f32, f32, f32) {
        if values.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let mut min_value = values[0];
        let mut max_value = values[0];
        let mut sum = 0.0f32;
        for value in values {
            min_value = min_value.min(*value);
            max_value = max_value.max(*value);
            sum += *value;
        }
        (min_value, max_value, sum / values.len() as f32)
    }

    #[cfg(feature = "runtime-model")]
    fn compare_tex_voxel_overlap(
        actual_coords: &[[u32; 4]],
        actual_attrs: &[[f32; 6]],
        reference_coords: &[[u32; 4]],
        reference_attrs: &[[f32; 6]],
    ) -> (
        crate::hook_diff::MetricStats,
        usize,
        usize,
        usize,
        [crate::hook_diff::MetricStats; 6],
    ) {
        let mut actual = HashMap::with_capacity(actual_coords.len().saturating_mul(2));
        for (coord, attr) in actual_coords
            .iter()
            .copied()
            .zip(actual_attrs.iter().copied())
        {
            actual.insert(coord, attr);
        }
        let mut reference = HashMap::with_capacity(reference_coords.len().saturating_mul(2));
        for (coord, attr) in reference_coords
            .iter()
            .copied()
            .zip(reference_attrs.iter().copied())
        {
            reference.insert(coord, attr);
        }

        let mut actual_flat = Vec::new();
        let mut reference_flat = Vec::new();
        let mut actual_channels = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        let mut reference_channels = [
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ];
        for (coord, reference_attr) in &reference {
            if let Some(actual_attr) = actual.get(coord) {
                actual_flat.extend(actual_attr);
                reference_flat.extend(reference_attr);
                for channel in 0..6 {
                    actual_channels[channel].push(actual_attr[channel]);
                    reference_channels[channel].push(reference_attr[channel]);
                }
            }
        }
        let overlap = actual_flat.len() / 6;
        let stats = compute_stats(actual_flat.as_slice(), reference_flat.as_slice());
        let per_channel = [
            compute_stats(
                actual_channels[0].as_slice(),
                reference_channels[0].as_slice(),
            ),
            compute_stats(
                actual_channels[1].as_slice(),
                reference_channels[1].as_slice(),
            ),
            compute_stats(
                actual_channels[2].as_slice(),
                reference_channels[2].as_slice(),
            ),
            compute_stats(
                actual_channels[3].as_slice(),
                reference_channels[3].as_slice(),
            ),
            compute_stats(
                actual_channels[4].as_slice(),
                reference_channels[4].as_slice(),
            ),
            compute_stats(
                actual_channels[5].as_slice(),
                reference_channels[5].as_slice(),
            ),
        ];
        (stats, overlap, actual.len(), reference.len(), per_channel)
    }
}

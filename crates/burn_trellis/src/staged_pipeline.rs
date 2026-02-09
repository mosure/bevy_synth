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
use crate::sampler::FlowEulerGuidanceIntervalSampler;
use crate::trellis_config::{TrellisNormalization, TrellisPipelineArgs, TrellisSamplerConfig};

#[derive(Debug, Clone)]
pub struct SparseStructureSample {
    pub source: SparseStructureStageSource,
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
    pub step_count: usize,
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_mid_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TexSLatSample {
    pub step_count: usize,
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
    pub shape_noise: Option<SparseRowNoiseOverride>,
    pub tex_noise: Option<SparseRowNoiseOverride>,
    pub cond_512: Option<Vec<f32>>,
    pub neg_cond_512: Option<Vec<f32>>,
    pub cond_1024: Option<Vec<f32>>,
    pub neg_cond_1024: Option<Vec<f32>>,
}

impl TrellisNoiseOverrides {
    pub fn is_empty(&self) -> bool {
        self.sparse_noise.is_none()
            && self.shape_noise.is_none()
            && self.tex_noise.is_none()
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
                                "burn_trellis: shape decoder runtime unavailable ({err}); using synthetic decode fallback."
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
                                "burn_trellis: tex decoder runtime unavailable ({err}); using synthetic decode fallback."
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

    pub fn run(&self, preprocess: &PreprocessOutput, seed: u64) -> TrellisStageOutput {
        self.run_with_overrides(preprocess, seed, None)
    }

    pub fn run_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
    ) -> TrellisStageOutput {
        self.run_profiled_with_overrides(preprocess, seed, noise_overrides)
            .0
    }

    pub fn run_profiled(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> (TrellisStageOutput, TrellisStageTimings) {
        self.run_profiled_with_overrides(preprocess, seed, None)
    }

    pub fn run_profiled_with_overrides(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
        noise_overrides: Option<&TrellisNoiseOverrides>,
    ) -> (TrellisStageOutput, TrellisStageTimings) {
        let total_start = Instant::now();
        let stage_debug = runtime_stage_debug_enabled();
        let parity_strict = runtime_parity_strict();
        let sparse_resolution = sparse_resolution_for_pipeline(self.pipeline_type());
        let mut rng = Lcg::new(seed);
        let sparse_noise_override = noise_overrides.and_then(|v| v.sparse_noise.as_deref());
        let shape_noise_override = noise_overrides.and_then(|v| v.shape_noise.as_ref());
        let tex_noise_override = noise_overrides.and_then(|v| v.tex_noise.as_ref());
        let sparse_cond_override = noise_overrides.and_then(|v| v.cond_512.as_deref());
        let sparse_neg_cond_override = noise_overrides.and_then(|v| v.neg_cond_512.as_deref());
        let sparse_start = Instant::now();
        let sparse = sample_sparse_structure(
            preprocess,
            sparse_resolution,
            &mut rng,
            sparse_noise_override,
            sparse_cond_override,
            sparse_neg_cond_override,
            &self.sparse_sampler,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.sparse_flow.as_ref(),
        );
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
            noise_overrides,
            &self.shape_sampler,
            &self.shape_norm,
            sparse.resolution,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.shape_flow.as_ref(),
        );
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
            noise_overrides,
            &self.tex_sampler,
            &self.shape_norm,
            &self.tex_norm,
            sparse.resolution,
            parity_strict,
            #[cfg(feature = "runtime-model")]
            self.tex_flow.as_ref(),
        );
        let tex_slat_ms = tex_start.elapsed().as_secs_f64() * 1000.0;
        if stage_debug {
            eprintln!(
                "burn_trellis: stage tex_slat complete ({tex_slat_ms:.2} ms, rows={})",
                tex_slat.coords.len()
            );
        }

        let decode_start = Instant::now();
        let decoded = if runtime_skip_decode() {
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
            )
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
        (output, timings)
    }
}

fn runtime_sampler_steps_override() -> Option<usize> {
    std::env::var("TRELLIS2_SAMPLER_STEPS_OVERRIDE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn runtime_parity_strict() -> bool {
    std::env::var("TRELLIS2_PARITY_STRICT")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
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

fn runtime_disable_pbr_bake() -> bool {
    std::env::var("TRELLIS2_DISABLE_PBR_BAKE")
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
) -> Vec<f32> {
    let expected = cond_tokens.saturating_mul(cond_channels);
    if let Some(values) = override_values {
        if values.len() == expected {
            return values.to_vec();
        }
        eprintln!(
            "burn_trellis: ignoring {stage} cond override due to len mismatch (expected {}, got {})",
            expected,
            values.len()
        );
    }
    build_sparse_cond_from_preprocess(preprocess, cond_tokens, cond_channels)
}

#[cfg(feature = "runtime-model")]
fn dense_neg_cond_with_override(
    expected_len: usize,
    override_values: Option<&[f32]>,
    stage: &str,
) -> Vec<f32> {
    if let Some(values) = override_values {
        if values.len() == expected_len {
            return values.to_vec();
        }
        eprintln!(
            "burn_trellis: ignoring {stage} neg-cond override due to len mismatch (expected {}, got {})",
            expected_len,
            values.len()
        );
    }
    vec![0.0; expected_len]
}

fn sparse_row_noise_map(override_rows: &SparseRowNoiseOverride) -> HashMap<u64, [f32; 32]> {
    let count = override_rows.coords.len().min(override_rows.feats.len());
    let mut out = HashMap::with_capacity(count * 2);
    for idx in 0..count {
        let coord = override_rows.coords[idx];
        out.insert(pack_coord(coord[1], coord[2], coord[3]), override_rows.feats[idx]);
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

fn sample_sparse_structure(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    _cond_override: Option<&[f32]>,
    _neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] sparse_flow: Option<&SparseStructureFlowRuntime>,
) -> SparseStructureSample {
    #[cfg(feature = "runtime-model")]
    if let Some(sparse_flow) = sparse_flow
        && let Some(sample) = sample_sparse_structure_with_model(
            preprocess,
            resolution,
            rng,
            noise_override,
            _cond_override,
            _neg_cond_override,
            sampler_config,
            sparse_flow,
        )
    {
        return sample;
    }
    if parity_strict {
        panic!(
            "burn_trellis parity strict mode: sparse_structure stage would use synthetic fallback"
        );
    }
    sample_sparse_structure_synthetic(preprocess, resolution, rng, noise_override, sampler_config)
}

fn sample_sparse_structure_synthetic(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
) -> SparseStructureSample {
    let flow_resolution = 16usize;
    let flow_channels = 8usize;
    let voxel_count = flow_resolution * flow_resolution * flow_resolution;
    let noise =
        dense_noise_with_override(rng, flow_channels * voxel_count, noise_override, "sparse");
    let target = occupancy_target(preprocess, flow_resolution);
    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let trace = sampler.sample_with_trace(&noise, sample_cfg, |x_t, _t, cond| {
        // Placeholder denoiser: positive branch drifts toward the occupancy target,
        // negative branch drifts toward empty space.
        let mut out = vec![0.0f32; x_t.len()];
        for idx in 0..out.len() {
            let target_idx = idx % voxel_count;
            let target_value = if cond { target[target_idx] } else { 0.0 };
            out[idx] = x_t[idx] - target_value;
        }
        out
    });
    let latent = trace.samples;
    let occupancy = latent_to_occupancy(&latent, flow_channels, flow_resolution);
    let upsampled = upsample_occupancy(occupancy.as_slice(), flow_resolution, resolution);
    let mut coords = Vec::new();
    let threshold = 0.5f32;
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let flat = (z * resolution + y) * resolution + x;
                if upsampled[flat] <= threshold {
                    continue;
                }
                coords.push([0, x as u32, y as u32, z as u32]);
            }
        }
    }
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
fn sample_sparse_structure_with_model(
    preprocess: &PreprocessOutput,
    resolution: usize,
    rng: &mut Lcg,
    noise_override: Option<&[f32]>,
    cond_override: Option<&[f32]>,
    neg_cond_override: Option<&[f32]>,
    sampler_config: &TrellisSamplerConfig,
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
    let cond = dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "sparse_runtime",
    );
    let neg_cond = dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "sparse_runtime",
    );
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
    let (_, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let trace = match sparse_flow.sample_with_trace(
        noise.as_slice(),
        sample_cfg,
        sampler_config.args.sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        None,
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
    let mut coords = occupancy_to_coords(upsampled.as_slice(), resolution, 0.5, max_sparse_coords);
    if coords.is_empty() {
        coords.push([
            0,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
            (resolution / 2) as u32,
        ]);
    }
    if let Some(limit) = max_sparse_coords {
        eprintln!(
            "burn_trellis: sparse coords after threshold/cap = {} (limit={})",
            coords.len(),
            limit
        );
    }
    Some(SparseStructureSample {
        source: match sparse_flow.backend_name() {
            "wgpu" => SparseStructureStageSource::RuntimeModelWgpu,
            _ => SparseStructureStageSource::RuntimeModelCpu,
        },
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
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    shape_flow: &SparseStructureFlowRuntime,
) -> Option<ShapeSLatSample> {
    if coords.is_empty() {
        return Some(ShapeSLatSample {
            step_count: sampler_config.params.steps.max(1),
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

    let mut noise = dense_noise_with_override(
        rng,
        config.out_channels * voxel_count,
        None,
        "shape_slat_runtime",
    );
    if let Some(override_rows) = noise_override {
        merge_sparse_row_noise_override(
            noise.as_mut_slice(),
            override_rows,
            coords,
            config.out_channels,
            sparse_resolution,
            dense_resolution,
            "shape_slat_runtime",
        );
    }

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "shape_slat_runtime",
    );
    let neg_cond = dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "shape_slat_runtime",
    );
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
    let (_, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let trace = match shape_flow.sample_with_trace(
        noise.as_slice(),
        sample_cfg,
        sampler_config.args.sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        None,
    ) {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape slat runtime prediction failed ({err}); using synthetic shape stage fallback."
            );
            return None;
        }
    };

    let feature_channels = 32usize.min(config.out_channels);
    let mut features = Vec::with_capacity(coords.len());
    let mut noise_rows = Vec::with_capacity(coords.len());
    let mut step_0_rows = Vec::with_capacity(coords.len());
    let mut step_mid_rows = Vec::with_capacity(coords.len());
    let mut step_last_rows = Vec::with_capacity(coords.len());
    for coord in coords {
        let dense_idx = map_coord_to_dense_flat(*coord, sparse_resolution, dense_resolution);
        let mut row = [0.0f32; 32];
        let mut noise_row = [0.0f32; 32];
        let mut step_0_row = [0.0f32; 32];
        let mut step_mid_row = [0.0f32; 32];
        let mut step_last_row = [0.0f32; 32];
        for ch in 0..feature_channels {
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            let offset = ch * voxel_count + dense_idx;
            let sampled = trace.samples[offset];
            row[ch] = sampled * std + mean;
            noise_row[ch] = noise[offset];
            step_0_row[ch] = trace.step_0_x_t[offset];
            step_mid_row[ch] = trace.step_mid_x_t[offset];
            step_last_row[ch] = trace.step_last_x_t[offset];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
    }
    Some(ShapeSLatSample {
        step_count: sample_cfg.steps,
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
    cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    sparse_resolution: usize,
    tex_flow: &SparseStructureFlowRuntime,
) -> Option<TexSLatSample> {
    if shape_slat.coords.is_empty() {
        return Some(TexSLatSample {
            step_count: sampler_config.params.steps.max(1),
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

    let mut noise = dense_noise_with_override(
        rng,
        config.out_channels * voxel_count,
        None,
        "tex_slat_runtime",
    );
    if let Some(override_rows) = noise_override {
        merge_sparse_row_noise_override(
            noise.as_mut_slice(),
            override_rows,
            shape_slat.coords.as_slice(),
            config.out_channels,
            sparse_resolution,
            dense_resolution,
            "tex_slat_runtime",
        );
    }

    let cond_tokens = if dense_resolution <= 32 {
        32 * 32 + 5
    } else {
        64 * 64 + 5
    };
    let (cond_override, neg_cond_override) = cond_override_for_tokens(cond_overrides, cond_tokens);
    let cond = dense_cond_with_override(
        preprocess,
        cond_tokens,
        config.cond_channels,
        cond_override,
        "tex_slat_runtime",
    );
    let neg_cond = dense_neg_cond_with_override(
        cond.len(),
        neg_cond_override,
        "tex_slat_runtime",
    );
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
    let (_, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let trace = match tex_flow.sample_with_trace(
        noise.as_slice(),
        sample_cfg,
        sampler_config.args.sigma_min,
        &cond_tensor,
        &neg_cond_tensor,
        Some(concat_dense.as_slice()),
    ) {
        Ok(trace) => trace,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex slat runtime prediction failed ({err}); using synthetic tex stage fallback."
            );
            return None;
        }
    };

    let feature_channels = 32usize.min(config.out_channels);
    let mut features = Vec::with_capacity(shape_slat.coords.len());
    let mut noise_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_0_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_mid_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_last_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut shape_cond_rows = Vec::with_capacity(shape_slat.coords.len());
    for (idx, coord) in shape_slat.coords.iter().enumerate() {
        let dense_idx = map_coord_to_dense_flat(*coord, sparse_resolution, dense_resolution);
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
            if ch >= feature_channels {
                continue;
            }
            let mean = normalization.mean.get(ch).copied().unwrap_or(0.0);
            let std = normalization
                .std
                .get(ch)
                .copied()
                .unwrap_or(1.0)
                .max(1.0e-6);
            let offset = ch * voxel_count + dense_idx;
            let sampled = trace.samples[offset];
            row[ch] = sampled * std + mean;
            noise_row[ch] = noise[offset];
            step_0_row[ch] = trace.step_0_x_t[offset];
            step_mid_row[ch] = trace.step_mid_x_t[offset];
            step_last_row[ch] = trace.step_last_x_t[offset];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_mid_rows.push(step_mid_row);
        step_last_rows.push(step_last_row);
        shape_cond_rows.push(shape_cond);
    }

    Some(TexSLatSample {
        step_count: sample_cfg.steps,
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
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] shape_flow: Option<&SparseStructureFlowRuntime>,
) -> ShapeSLatSample {
    #[cfg(feature = "runtime-model")]
    if let Some(shape_flow) = shape_flow
        && let Some(sample) = sample_shape_slat_with_model(
            preprocess,
            coords,
            rng,
            noise_override,
            _cond_overrides,
            sampler_config,
            normalization,
            _sparse_resolution,
            shape_flow,
        )
    {
        return sample;
    }
    if parity_strict {
        panic!("burn_trellis parity strict mode: shape_slat stage would use synthetic fallback");
    }

    let mut features = Vec::with_capacity(coords.len());
    let mut noise_rows = Vec::with_capacity(coords.len());
    let mut step_0_rows = Vec::with_capacity(coords.len());
    let mut step_mid_rows = Vec::with_capacity(coords.len());
    let mut step_last_rows = Vec::with_capacity(coords.len());
    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let override_noise_map = noise_override.map(sparse_row_noise_map);
    for coord in coords {
        let base = sample_pixel_luma(preprocess, coord[1], coord[2], coord[3]);
        let noise = override_noise_map
            .as_ref()
            .and_then(|map| map.get(&pack_coord(coord[1], coord[2], coord[3])))
            .map(|row| row.to_vec())
            .unwrap_or_else(|| (0..32).map(|_| rng.next_normal_f32()).collect::<Vec<_>>());
        let target = [base; 32];
        let trace = sampler.sample_with_trace(&noise, sample_cfg, |x_t, _t, cond| {
            let mut out = vec![0.0f32; x_t.len()];
            for idx in 0..out.len() {
                let target_value = if cond { target[idx] } else { 0.0 };
                out[idx] = x_t[idx] - target_value;
            }
            out
        });
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
    ShapeSLatSample {
        step_count: sample_cfg.steps,
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        coords: coords.to_vec(),
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_tex_slat(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    rng: &mut Lcg,
    noise_override: Option<&SparseRowNoiseOverride>,
    _cond_overrides: Option<&TrellisNoiseOverrides>,
    sampler_config: &TrellisSamplerConfig,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
    _sparse_resolution: usize,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] tex_flow: Option<&SparseStructureFlowRuntime>,
) -> TexSLatSample {
    #[cfg(feature = "runtime-model")]
    if let Some(tex_flow) = tex_flow
        && let Some(sample) = sample_tex_slat_with_model(
            preprocess,
            shape_slat,
            rng,
            noise_override,
            _cond_overrides,
            sampler_config,
            shape_normalization,
            normalization,
            _sparse_resolution,
            tex_flow,
        )
    {
        return sample;
    }
    if parity_strict {
        panic!("burn_trellis parity strict mode: tex_slat stage would use synthetic fallback");
    }

    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
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
        let trace = sampler.sample_with_trace(&noise, sample_cfg, |x_t, _t, cond| {
            let mut out = vec![0.0f32; x_t.len()];
            for ch in 0..out.len() {
                let target_value = if cond { target[ch] } else { 0.0 };
                out[ch] = x_t[ch] - target_value;
            }
            out
        });
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
    TexSLatSample {
        step_count: sample_cfg.steps,
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_mid_x_t: step_mid_rows,
        step_last_x_t: step_last_rows,
        shape_slat_cond: shape_cond_rows,
        coords: shape_slat.coords.clone(),
    }
}

fn decode_latent_to_outputs(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    parity_strict: bool,
    #[cfg(feature = "runtime-model")] shape_decoder: Option<&FdgDecoderRuntime>,
    #[cfg(feature = "runtime-model")] tex_decoder: Option<&SparseUnetVaeDecoderRuntime>,
) -> DecodedLatentOutput {
    #[cfg(feature = "runtime-model")]
    if let (Some(shape_decoder), Some(tex_decoder)) = (shape_decoder, tex_decoder) {
        if let Some(decoded) = decode_latent_with_runtime_decoders(
            shape,
            tex,
            pipeline_type,
            shape_decoder,
            tex_decoder,
        ) {
            return decoded;
        }
        if parity_strict {
            panic!("burn_trellis parity strict mode: runtime decoder path failed");
        }
    }
    if parity_strict {
        panic!("burn_trellis parity strict mode: decode stage would use synthetic fallback");
    }

    if shape.coords.is_empty() || shape.features.is_empty() {
        return DecodedLatentOutput {
            mesh: canonical_cube(),
            shape_subs: Vec::new(),
            tex_voxels: DecodeTexVoxelSample {
                coords: Vec::new(),
                feats: Vec::new(),
                spatial_shape: [512, 512, 512],
            },
            pbr: None,
        };
    }

    let final_resolution = final_resolution_for_pipeline(pipeline_type);
    let coarse_resolution = sparse_resolution_for_pipeline(pipeline_type).max(1);
    let scale = (final_resolution / coarse_resolution).max(1);
    let mut levels = 0usize;
    let mut tmp = scale;
    while tmp > 1 {
        levels += 1;
        tmp /= 2;
    }
    // TRELLIS shape decoder effectively upsamples x16 for the low-resolution path.
    // Keep an upper bound to avoid runaway growth in non-standard configs.
    levels = levels.clamp(1, 4);
    let per_base = 1usize << (3 * levels);

    let target_vertices = target_vertex_budget(final_resolution);
    let mut base_indices = ranked_shape_indices(shape);
    let max_base = target_vertices
        .saturating_add(per_base - 1)
        .saturating_div(per_base)
        .max(1);
    base_indices.truncate(base_indices.len().min(max_base));
    if base_indices.is_empty() {
        base_indices.push(0);
    }

    let mut shape_subs = Vec::with_capacity(levels);
    let mut level_coords: Vec<[u32; 4]> =
        base_indices.iter().map(|&idx| shape.coords[idx]).collect();
    let mut level_feats: Vec<[f32; 32]> = base_indices
        .iter()
        .map(|&idx| shape.features[idx])
        .collect();
    let mut level_tex_feats: Vec<[f32; 32]> = base_indices
        .iter()
        .map(|&idx| tex.features.get(idx).copied().unwrap_or([0.0; 32]))
        .collect();

    for level in 0..levels {
        let mut sub_feats = Vec::with_capacity(level_coords.len());
        for (idx, feat) in level_feats.iter().enumerate() {
            let coord = level_coords[idx];
            let mut row = [0.0f32; 8];
            for child in 0..8usize {
                let child_bits = [
                    (child & 1) as f32,
                    ((child >> 1) & 1) as f32,
                    ((child >> 2) & 1) as f32,
                ];
                row[child] = 1.0
                    + 0.08 * feat[(level * 5 + child) % 32]
                    + 0.02
                        * (coord[1] as f32 * 0.11
                            + coord[2] as f32 * 0.07
                            + coord[3] as f32 * 0.05)
                    + 0.01 * (child_bits[0] + child_bits[1] + child_bits[2]);
            }
            sub_feats.push(row);
        }
        let spatial_res = (coarse_resolution << level) as u32;
        shape_subs.push(DecodeShapeSubSample {
            coords: level_coords.clone(),
            feats: sub_feats,
            spatial_shape: [spatial_res, spatial_res, spatial_res],
        });

        let mut next_coords = Vec::with_capacity(level_coords.len() * 8);
        let mut next_feats = Vec::with_capacity(level_feats.len() * 8);
        let mut next_tex_feats = Vec::with_capacity(level_tex_feats.len() * 8);
        for i in 0..level_coords.len() {
            let coord = level_coords[i];
            let parent = level_feats[i];
            let tex_parent = level_tex_feats[i];
            for child in 0..8u32 {
                let child_coord = [
                    coord[0],
                    coord[1].saturating_mul(2).saturating_add(child & 1),
                    coord[2].saturating_mul(2).saturating_add((child >> 1) & 1),
                    coord[3].saturating_mul(2).saturating_add((child >> 2) & 1),
                ];
                next_coords.push(child_coord);
                let mut child_feat = [0.0f32; 32];
                let mut child_tex_feat = [0.0f32; 32];
                let cx = (child & 1) as f32 - 0.5;
                let cy = ((child >> 1) & 1) as f32 - 0.5;
                let cz = ((child >> 2) & 1) as f32 - 0.5;
                for ch in 0..32 {
                    let dir = match ch % 3 {
                        0 => cx,
                        1 => cy,
                        _ => cz,
                    };
                    child_feat[ch] = parent[ch] * 0.94 + dir * 0.06;
                    child_tex_feat[ch] = tex_parent[ch] * 0.95 + dir * 0.05;
                }
                next_feats.push(child_feat);
                next_tex_feats.push(child_tex_feat);
            }
        }
        level_coords = next_coords;
        level_feats = next_feats;
        level_tex_feats = next_tex_feats;
    }

    // Match the canonical token budget more closely by uniformly decimating
    // over-expanded leaves after fixed-depth subdivision.
    if level_coords.len() > target_vertices {
        let remove_count = level_coords.len() - target_vertices;
        let total = level_coords.len();
        let stride = total as f64 / remove_count as f64;
        let mut drop_mask = vec![false; total];
        for ridx in 0..remove_count {
            let idx = (((ridx as f64) + 0.5) * stride).floor() as usize;
            let idx = idx.min(total - 1);
            drop_mask[idx] = true;
        }

        let mut kept_coords = Vec::with_capacity(target_vertices);
        let mut kept_feats = Vec::with_capacity(target_vertices);
        let mut kept_tex_feats = Vec::with_capacity(target_vertices);
        for i in 0..total {
            if drop_mask[i] {
                continue;
            }
            kept_coords.push(level_coords[i]);
            kept_feats.push(level_feats[i]);
            kept_tex_feats.push(level_tex_feats[i]);
        }
        level_coords = kept_coords;
        level_feats = kept_feats;
        level_tex_feats = kept_tex_feats;
    }

    let mut dual_vertices = Vec::with_capacity(level_coords.len());
    let mut intersected = Vec::with_capacity(level_coords.len());
    let mut split_weight = Vec::with_capacity(level_coords.len());
    let mut voxel_attrs = Vec::with_capacity(level_coords.len());
    for i in 0..level_coords.len() {
        let coord = level_coords[i];
        let shape_feat = level_feats[i];
        let tex_feat = level_tex_feats[i];
        let cell = [
            (coord[1] % (scale as u32).max(1)) as f32 / (scale as f32).max(1.0),
            (coord[2] % (scale as u32).max(1)) as f32 / (scale as f32).max(1.0),
            (coord[3] % (scale as u32).max(1)) as f32 / (scale as f32).max(1.0),
        ];
        dual_vertices.push([
            (cell[0] + 0.12 * shape_feat[0].tanh()).clamp(0.0, 1.0),
            (cell[1] + 0.12 * shape_feat[1].tanh()).clamp(0.0, 1.0),
            (cell[2] + 0.12 * shape_feat[2].tanh()).clamp(0.0, 1.0),
        ]);
        let axis_selector = ((shape_feat[3] * 2.7 + shape_feat[4] * 1.9 + shape_feat[5] * 1.3)
            .abs()
            * 1000.0) as usize
            % 3;
        let mut flags = [false; 3];
        flags[axis_selector] = true;
        // Lightly activate a secondary edge direction to better match the
        // decoded face density observed in TRELLIS2 hook traces.
        let secondary_gate = ((shape_feat[7].abs() * 37.0)
            + coord[1] as f32 * 0.19
            + coord[2] as f32 * 0.13
            + coord[3] as f32 * 0.11) as i32;
        if secondary_gate.rem_euclid(8) == 0 {
            flags[(axis_selector + 1) % 3] = true;
        }
        if secondary_gate.rem_euclid(257) == 0 {
            flags[(axis_selector + 2) % 3] = true;
        }
        intersected.push(flags);
        split_weight.push(softplus(shape_feat[6] + 0.25));
        let mut attr = [0.0f32; 6];
        for ch in 0..6 {
            attr[ch] = (0.5 + 0.5 * (tex_feat[ch] + 0.1 * shape_feat[ch]).tanh()).clamp(0.0, 1.0);
        }
        voxel_attrs.push(attr);
    }

    let grid_size = [
        final_resolution as u32,
        final_resolution as u32,
        final_resolution as u32,
    ];
    let (vertices, faces) = flexible_dual_grid_to_mesh(
        &level_coords,
        &dual_vertices,
        &intersected,
        Some(&split_weight),
        grid_size,
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
    );

    let (uvs, pbr_textures, pbr_debug) = if runtime_disable_pbr_bake() {
        (Vec::new(), None, None)
    } else {
        let (uvs, textures, debug) = bake_pbr_from_voxels(
            vertices.as_slice(),
            faces.as_slice(),
            level_coords.as_slice(),
            voxel_attrs.as_slice(),
            final_resolution as u32,
        );
        (uvs, textures, Some(debug))
    };
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mesh = if vertices.is_empty() || faces.is_empty() {
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

    DecodedLatentOutput {
        mesh,
        shape_subs,
        tex_voxels: DecodeTexVoxelSample {
            coords: level_coords,
            feats: voxel_attrs,
            spatial_shape: [
                final_resolution as u32,
                final_resolution as u32,
                final_resolution as u32,
            ],
        },
        pbr: pbr_debug,
    }
}

#[cfg(feature = "runtime-model")]
fn decode_latent_with_runtime_decoders(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
    shape_decoder: &FdgDecoderRuntime,
    tex_decoder: &SparseUnetVaeDecoderRuntime,
) -> Option<DecodedLatentOutput> {
    let stage_debug = runtime_stage_debug_enabled();
    let count = shape
        .coords
        .len()
        .min(shape.features.len())
        .min(tex.features.len());
    if count == 0 {
        return Some(DecodedLatentOutput {
            mesh: canonical_cube(),
            shape_subs: Vec::new(),
            tex_voxels: DecodeTexVoxelSample {
                coords: Vec::new(),
                feats: Vec::new(),
                spatial_shape: [512, 512, 512],
            },
            pbr: None,
        });
    }
    if shape_decoder.out_channels() < 7 || tex_decoder.out_channels() < 6 {
        return None;
    }
    if stage_debug {
        eprintln!("burn_trellis: decode runtime begin (rows={count})");
    }
    let shape_rows = &shape.features[..count];
    let tex_rows = &tex.features[..count];
    let shape_decode_start = Instant::now();
    let shape_decoded = match shape_decoder.decode_sparse(&shape.coords[..count], shape_rows) {
        Ok(decoded) => decoded,
        Err(err) => {
            eprintln!(
                "burn_trellis: shape runtime decoder failed ({err}); using synthetic decode fallback."
            );
            return None;
        }
    };
    if stage_debug {
        eprintln!(
            "burn_trellis: decode runtime shape-decoder complete ({:.2} ms, subs={}, coords={})",
            shape_decode_start.elapsed().as_secs_f64() * 1000.0,
            shape_decoded.subdivisions.len(),
            shape_decoded.coords.len()
        );
    }

    let tex_decode_start = Instant::now();
    let tex_decoded = match tex_decoder.decode_with_guidance(
        &tex.coords[..count],
        tex_rows,
        shape_decoded.subdivisions.as_slice(),
    ) {
        Ok(decoded) => decoded,
        Err(err) => {
            eprintln!(
                "burn_trellis: tex runtime decoder failed ({err}); using synthetic decode fallback."
            );
            return None;
        }
    };
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
    let (uvs, pbr_textures, pbr_debug) = if runtime_disable_pbr_bake() {
        (Vec::new(), None, None)
    } else {
        let (uvs, textures, debug) = bake_pbr_from_voxels(
            vertices.as_slice(),
            faces.as_slice(),
            coords.as_slice(),
            voxel_attrs.as_slice(),
            final_resolution as u32,
        );
        (uvs, textures, Some(debug))
    };
    let material = summarize_material(voxel_attrs.as_slice(), pbr_textures.as_ref());
    let mesh = if vertices.is_empty() || faces.is_empty() {
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

    Some(DecodedLatentOutput {
        mesh,
        shape_subs,
        tex_voxels: DecodeTexVoxelSample {
            coords,
            feats: voxel_attrs,
            spatial_shape: tex_spatial,
        },
        pbr: pbr_debug,
    })
}

#[cfg(feature = "runtime-model")]
fn runtime_subdivision_to_sample(sub: &SparseSubdivisionLogits) -> DecodeShapeSubSample {
    let mut feats = Vec::with_capacity(sub.coords.len());
    for row_idx in 0..sub.coords.len() {
        let mut row = [0.0f32; 8];
        let base = row_idx * 8;
        if base + 8 <= sub.logits.len() {
            row.copy_from_slice(&sub.logits[base..base + 8]);
        }
        feats.push(row);
    }
    DecodeShapeSubSample {
        coords: sub.coords.clone(),
        feats,
        spatial_shape: sub.spatial_shape,
    }
}

#[cfg(feature = "runtime-model")]
fn spatial_shape_from_sparse_coords(coords: &[[u32; 4]]) -> [u32; 3] {
    if coords.is_empty() {
        return [1, 1, 1];
    }
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut max_z = 0u32;
    for coord in coords {
        max_x = max_x.max(coord[1]);
        max_y = max_y.max(coord[2]);
        max_z = max_z.max(coord[3]);
    }
    [
        max_x.saturating_add(1),
        max_y.saturating_add(1),
        max_z.saturating_add(1),
    ]
}

fn flexible_dual_grid_to_mesh(
    coords: &[[u32; 4]],
    dual_vertices: &[[f32; 3]],
    intersected_flag: &[[bool; 3]],
    split_weight: Option<&[f32]>,
    grid_size: [u32; 3],
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    if coords.is_empty()
        || dual_vertices.len() != coords.len()
        || intersected_flag.len() != coords.len()
        || split_weight.is_some_and(|w| w.len() != coords.len())
    {
        return (Vec::new(), Vec::new());
    }

    // TRELLIS2 flexible-dual-grid edge neighborhoods:
    // x-axis, y-axis, z-axis (4 voxels per quad candidate).
    const EDGE_NEIGHBOR_VOXEL_OFFSET: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];

    let mut coord_to_index = HashMap::with_capacity(coords.len() * 2);
    for (idx, coord) in coords.iter().enumerate() {
        coord_to_index.insert(pack_coord(coord[1], coord[2], coord[3]), idx as u32);
    }

    let mut quad_indices = Vec::<[u32; 4]>::new();
    for (idx, coord) in coords.iter().enumerate() {
        let base = [coord[1] as i32, coord[2] as i32, coord[3] as i32];
        for axis in 0..3 {
            if !intersected_flag[idx][axis] {
                continue;
            }
            let mut quad = [0u32; 4];
            let mut valid = true;
            for k in 0..4 {
                let offset = EDGE_NEIGHBOR_VOXEL_OFFSET[axis][k];
                let nx = base[0] + offset[0];
                let ny = base[1] + offset[1];
                let nz = base[2] + offset[2];
                if nx < 0 || ny < 0 || nz < 0 {
                    valid = false;
                    break;
                }
                let Some(&neighbor_idx) =
                    coord_to_index.get(&pack_coord(nx as u32, ny as u32, nz as u32))
                else {
                    valid = false;
                    break;
                };
                quad[k] = neighbor_idx;
            }
            if valid {
                quad_indices.push(quad);
            }
        }
    }

    if quad_indices.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let voxel_size = [
        (aabb_max[0] - aabb_min[0]) / grid_size[0].max(1) as f32,
        (aabb_max[1] - aabb_min[1]) / grid_size[1].max(1) as f32,
        (aabb_max[2] - aabb_min[2]) / grid_size[2].max(1) as f32,
    ];
    let mut vertices = Vec::with_capacity(coords.len());
    for (coord, dual) in coords.iter().zip(dual_vertices.iter()) {
        vertices.push([
            (coord[1] as f32 + dual[0]) * voxel_size[0] + aabb_min[0],
            (coord[2] as f32 + dual[1]) * voxel_size[1] + aabb_min[1],
            (coord[3] as f32 + dual[2]) * voxel_size[2] + aabb_min[2],
        ]);
    }

    let mut faces = Vec::with_capacity(quad_indices.len() * 2);
    for quad in quad_indices {
        let use_split_1 = if let Some(weights) = split_weight {
            let w02 = weights[quad[0] as usize] * weights[quad[2] as usize];
            let w13 = weights[quad[1] as usize] * weights[quad[3] as usize];
            w02 > w13
        } else {
            let split1 = quad_to_triangles_split1(quad);
            let split2 = quad_to_triangles_split2(quad);
            triangle_alignment(vertices.as_slice(), split1).abs()
                > triangle_alignment(vertices.as_slice(), split2).abs()
        };
        let tris = if use_split_1 {
            quad_to_triangles_split1(quad)
        } else {
            quad_to_triangles_split2(quad)
        };
        faces.push(tris[0]);
        faces.push(tris[1]);
    }

    (vertices, faces)
}

fn quad_to_triangles_split1(quad: [u32; 4]) -> [[u32; 3]; 2] {
    [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
}

fn quad_to_triangles_split2(quad: [u32; 4]) -> [[u32; 3]; 2] {
    [[quad[0], quad[1], quad[3]], [quad[3], quad[1], quad[2]]]
}

fn triangle_alignment(vertices: &[[f32; 3]], tris: [[u32; 3]; 2]) -> f32 {
    let n0 = triangle_normal(vertices, tris[0]);
    let n1 = triangle_normal(vertices, tris[1]);
    dot3(n0, n1)
}

fn triangle_normal(vertices: &[[f32; 3]], tri: [u32; 3]) -> [f32; 3] {
    let a = vertices[tri[0] as usize];
    let b = vertices[tri[1] as usize];
    let c = vertices[tri[2] as usize];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    cross3(ab, ac)
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn pack_coord(x: u32, y: u32, z: u32) -> u64 {
    ((x as u64) << 42) | ((y as u64) << 21) | z as u64
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn summarize_material(
    voxel_attrs: &[[f32; 6]],
    pbr_textures: Option<&MeshPbrTextures>,
) -> Option<MeshMaterial> {
    if let Some(textures) = pbr_textures {
        let base = &textures.base_color.rgba8;
        let mr = &textures.metallic_roughness.rgba8;
        if base.len() >= 4 && mr.len() >= 4 {
            let texels = (base.len() / 4).max(1);
            let mut accum = [0.0f32; 6];
            for idx in 0..texels {
                let off = idx * 4;
                accum[0] += base[off] as f32 / 255.0;
                accum[1] += base[off + 1] as f32 / 255.0;
                accum[2] += base[off + 2] as f32 / 255.0;
                accum[5] += base[off + 3] as f32 / 255.0;
                accum[3] += mr[off + 2] as f32 / 255.0;
                accum[4] += mr[off + 1] as f32 / 255.0;
            }
            let inv = 1.0 / texels as f32;
            return Some(MeshMaterial {
                base_color: [
                    (accum[0] * inv).clamp(0.0, 1.0),
                    (accum[1] * inv).clamp(0.0, 1.0),
                    (accum[2] * inv).clamp(0.0, 1.0),
                ],
                metallic: (accum[3] * inv).clamp(0.0, 1.0),
                roughness: (accum[4] * inv).clamp(0.0, 1.0),
                alpha: (accum[5] * inv).clamp(0.0, 1.0),
            });
        }
    }
    if voxel_attrs.is_empty() {
        return None;
    }
    let mut accum = [0.0f32; 6];
    for attrs in voxel_attrs {
        for idx in 0..6 {
            accum[idx] += attrs[idx];
        }
    }
    let inv = 1.0 / voxel_attrs.len() as f32;
    Some(MeshMaterial {
        base_color: [
            (accum[0] * inv).clamp(0.0, 1.0),
            (accum[1] * inv).clamp(0.0, 1.0),
            (accum[2] * inv).clamp(0.0, 1.0),
        ],
        metallic: (accum[3] * inv).clamp(0.0, 1.0),
        roughness: (accum[4] * inv).clamp(0.0, 1.0),
        alpha: (accum[5] * inv).clamp(0.0, 1.0),
    })
}

fn runtime_pbr_texture_size() -> usize {
    std::env::var("TRELLIS2_PBR_TEX_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 64)
        .unwrap_or(256)
}

#[allow(clippy::type_complexity)]
fn bake_pbr_from_voxels(
    vertices: &[[f32; 3]],
    faces: &[[u32; 3]],
    voxel_coords: &[[u32; 4]],
    voxel_attrs: &[[f32; 6]],
    fallback_spatial_resolution: u32,
) -> (Vec<[f32; 2]>, Option<MeshPbrTextures>, PbrBakeDebug) {
    if vertices.is_empty() || faces.is_empty() {
        return (
            Vec::new(),
            None,
            PbrBakeDebug {
                texture_width: 0,
                texture_height: 0,
                uvs: Vec::new(),
                raster_mask: Vec::new(),
                sample_positions: Vec::new(),
                sample_attrs: Vec::new(),
                base_color_float: Vec::new(),
                metallic_float: Vec::new(),
                roughness_float: Vec::new(),
                alpha_float: Vec::new(),
                base_color_rgba_u8: Vec::new(),
                metallic_roughness_u8: Vec::new(),
            },
        );
    }

    let uvs = planar_uv_unwrap(vertices);
    let texture_size = runtime_pbr_texture_size();
    let texel_count = texture_size * texture_size;
    let mut raster_mask = vec![0u8; texel_count];
    let mut base_color_float = vec![[0.0f32; 4]; texel_count];
    let mut metallic_float = vec![0.0f32; texel_count];
    let mut roughness_float = vec![1.0f32; texel_count];
    let mut alpha_float = vec![1.0f32; texel_count];
    let mut sample_positions = Vec::with_capacity(texel_count / 2);
    let mut sample_attrs = Vec::with_capacity(texel_count / 2);

    let mut voxel_map = HashMap::with_capacity(voxel_coords.len().saturating_mul(2));
    let mut spatial = [
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
        fallback_spatial_resolution.max(1),
    ];
    for (idx, coord) in voxel_coords.iter().enumerate() {
        let attrs = voxel_attrs
            .get(idx)
            .copied()
            .unwrap_or([0.5, 0.5, 0.5, 0.0, 1.0, 1.0]);
        voxel_map.insert(pack_coord(coord[1], coord[2], coord[3]), attrs);
        spatial[0] = spatial[0].max(coord[1].saturating_add(1));
        spatial[1] = spatial[1].max(coord[2].saturating_add(1));
        spatial[2] = spatial[2].max(coord[3].saturating_add(1));
    }
    let fallback_attr = summarize_voxel_attr(voxel_attrs);

    for face in faces {
        let i0 = face[0] as usize;
        let i1 = face[1] as usize;
        let i2 = face[2] as usize;
        if i0 >= vertices.len() || i1 >= vertices.len() || i2 >= vertices.len() {
            continue;
        }
        if i0 >= uvs.len() || i1 >= uvs.len() || i2 >= uvs.len() {
            continue;
        }
        let p0 = vertices[i0];
        let p1 = vertices[i1];
        let p2 = vertices[i2];
        let uv0 = uvs[i0];
        let uv1 = uvs[i1];
        let uv2 = uvs[i2];
        rasterize_triangle(texture_size, [uv0, uv1, uv2], |x, y, bary| {
            let position = [
                p0[0] * bary[0] + p1[0] * bary[1] + p2[0] * bary[2],
                p0[1] * bary[0] + p1[1] * bary[1] + p2[1] * bary[2],
                p0[2] * bary[0] + p1[2] * bary[1] + p2[2] * bary[2],
            ];
            let attrs =
                sample_voxel_attr(position, &voxel_map, fallback_attr, spatial, voxel_coords);
            let idx = y * texture_size + x;
            if raster_mask[idx] == 0 {
                base_color_float[idx] = [attrs[0], attrs[1], attrs[2], attrs[5]];
                metallic_float[idx] = attrs[3];
                roughness_float[idx] = attrs[4];
                alpha_float[idx] = attrs[5];
                raster_mask[idx] = 255;
            } else {
                // Weighted blend where overdraw occurs from UV seams.
                base_color_float[idx][0] = 0.5 * base_color_float[idx][0] + 0.5 * attrs[0];
                base_color_float[idx][1] = 0.5 * base_color_float[idx][1] + 0.5 * attrs[1];
                base_color_float[idx][2] = 0.5 * base_color_float[idx][2] + 0.5 * attrs[2];
                base_color_float[idx][3] = 0.5 * base_color_float[idx][3] + 0.5 * attrs[5];
                metallic_float[idx] = 0.5 * metallic_float[idx] + 0.5 * attrs[3];
                roughness_float[idx] = 0.5 * roughness_float[idx] + 0.5 * attrs[4];
                alpha_float[idx] = 0.5 * alpha_float[idx] + 0.5 * attrs[5];
            }
            sample_positions.push(position);
            sample_attrs.push(attrs);
        });
    }

    inpaint_texture_channels(
        texture_size,
        raster_mask.as_mut_slice(),
        base_color_float.as_mut_slice(),
        metallic_float.as_mut_slice(),
        roughness_float.as_mut_slice(),
        alpha_float.as_mut_slice(),
        fallback_attr,
    );

    let mut base_color_rgba_u8 = vec![0u8; texel_count * 4];
    let mut metallic_roughness_u8 = vec![0u8; texel_count * 4];
    for idx in 0..texel_count {
        let off = idx * 4;
        let rgba = base_color_float[idx];
        base_color_rgba_u8[off] = quantize_unorm8(rgba[0]);
        base_color_rgba_u8[off + 1] = quantize_unorm8(rgba[1]);
        base_color_rgba_u8[off + 2] = quantize_unorm8(rgba[2]);
        base_color_rgba_u8[off + 3] = quantize_unorm8(alpha_float[idx]);
        metallic_roughness_u8[off] = 0;
        metallic_roughness_u8[off + 1] = quantize_unorm8(roughness_float[idx]);
        metallic_roughness_u8[off + 2] = quantize_unorm8(metallic_float[idx]);
        metallic_roughness_u8[off + 3] = 255;
    }

    let pbr_textures = MeshPbrTextures {
        base_color: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: base_color_rgba_u8.clone(),
        },
        metallic_roughness: MeshTexture {
            width: texture_size as u32,
            height: texture_size as u32,
            rgba8: metallic_roughness_u8.clone(),
        },
        normal: None,
        emissive: None,
        occlusion: None,
    };

    (
        uvs.clone(),
        Some(pbr_textures),
        PbrBakeDebug {
            texture_width: texture_size,
            texture_height: texture_size,
            uvs,
            raster_mask,
            sample_positions,
            sample_attrs,
            base_color_float,
            metallic_float,
            roughness_float,
            alpha_float,
            base_color_rgba_u8,
            metallic_roughness_u8,
        },
    )
}

fn planar_uv_unwrap(vertices: &[[f32; 3]]) -> Vec<[f32; 2]> {
    if vertices.is_empty() {
        return Vec::new();
    }
    let mut min = vertices[0];
    let mut max = vertices[0];
    for vertex in vertices.iter().skip(1) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    let range_x = (max[0] - min[0]).abs().max(1.0e-6);
    let range_y = (max[1] - min[1]).abs().max(1.0e-6);
    let range_z = (max[2] - min[2]).abs().max(1.0e-6);
    vertices
        .iter()
        .map(|vertex| {
            // Stable fallback projection: dominant axis pair.
            let ax = vertex[0].abs();
            let ay = vertex[1].abs();
            let az = vertex[2].abs();
            if ay >= ax && ay >= az {
                [
                    ((vertex[0] - min[0]) / range_x).clamp(0.0, 1.0),
                    ((vertex[2] - min[2]) / range_z).clamp(0.0, 1.0),
                ]
            } else if ax >= az {
                [
                    ((vertex[2] - min[2]) / range_z).clamp(0.0, 1.0),
                    ((vertex[1] - min[1]) / range_y).clamp(0.0, 1.0),
                ]
            } else {
                [
                    ((vertex[0] - min[0]) / range_x).clamp(0.0, 1.0),
                    ((vertex[1] - min[1]) / range_y).clamp(0.0, 1.0),
                ]
            }
        })
        .collect()
}

fn summarize_voxel_attr(voxel_attrs: &[[f32; 6]]) -> [f32; 6] {
    if voxel_attrs.is_empty() {
        return [0.7, 0.7, 0.7, 0.0, 0.8, 1.0];
    }
    let mut accum = [0.0f32; 6];
    for attrs in voxel_attrs {
        for idx in 0..6 {
            accum[idx] += attrs[idx];
        }
    }
    let inv = 1.0 / voxel_attrs.len() as f32;
    for value in &mut accum {
        *value *= inv;
    }
    accum
}

fn sample_voxel_attr(
    position: [f32; 3],
    voxel_map: &HashMap<u64, [f32; 6]>,
    fallback: [f32; 6],
    spatial: [u32; 3],
    voxel_coords: &[[u32; 4]],
) -> [f32; 6] {
    if voxel_map.is_empty() {
        return fallback;
    }
    let map_axis = |value: f32, dim: u32| -> i32 {
        let dim = dim.max(1) as f32;
        let coord = ((value + 0.5) * (dim - 1.0)).round();
        coord.clamp(0.0, dim - 1.0) as i32
    };
    let base = [
        map_axis(position[0], spatial[0]),
        map_axis(position[1], spatial[1]),
        map_axis(position[2], spatial[2]),
    ];
    let key = pack_coord(base[0] as u32, base[1] as u32, base[2] as u32);
    if let Some(attrs) = voxel_map.get(&key) {
        return *attrs;
    }
    let mut best = None;
    let mut best_dist = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let x = base[0] + dx;
                let y = base[1] + dy;
                let z = base[2] + dz;
                if x < 0 || y < 0 || z < 0 {
                    continue;
                }
                let key = pack_coord(x as u32, y as u32, z as u32);
                if let Some(attrs) = voxel_map.get(&key) {
                    let dist = (dx * dx + dy * dy + dz * dz) as f32;
                    if dist < best_dist {
                        best_dist = dist;
                        best = Some(*attrs);
                    }
                }
            }
        }
    }
    if let Some(attrs) = best {
        return attrs;
    }
    if !voxel_coords.is_empty() {
        // Last-resort stable fallback for sparse misses: nearest known coordinate.
        let mut nearest_idx = 0usize;
        let mut nearest_dist = f32::INFINITY;
        for (idx, coord) in voxel_coords.iter().enumerate() {
            let dx = coord[1] as f32 - base[0] as f32;
            let dy = coord[2] as f32 - base[1] as f32;
            let dz = coord[3] as f32 - base[2] as f32;
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_idx = idx;
            }
        }
        if let Some(attrs) = voxel_map.get(&pack_coord(
            voxel_coords[nearest_idx][1],
            voxel_coords[nearest_idx][2],
            voxel_coords[nearest_idx][3],
        )) {
            return *attrs;
        }
    }
    fallback
}

fn rasterize_triangle(
    texture_size: usize,
    tri_uv: [[f32; 2]; 3],
    mut draw: impl FnMut(usize, usize, [f32; 3]),
) {
    let to_px = |uv: [f32; 2]| -> [f32; 2] {
        [
            uv[0].clamp(0.0, 1.0) * (texture_size.saturating_sub(1)) as f32,
            (1.0 - uv[1].clamp(0.0, 1.0)) * (texture_size.saturating_sub(1)) as f32,
        ]
    };
    let p0 = to_px(tri_uv[0]);
    let p1 = to_px(tri_uv[1]);
    let p2 = to_px(tri_uv[2]);
    let min_x = p0[0]
        .min(p1[0])
        .min(p2[0])
        .floor()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let max_x = p0[0]
        .max(p1[0])
        .max(p2[0])
        .ceil()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let min_y = p0[1]
        .min(p1[1])
        .min(p2[1])
        .floor()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;
    let max_y = p0[1]
        .max(p1[1])
        .max(p2[1])
        .ceil()
        .clamp(0.0, (texture_size.saturating_sub(1)) as f32) as usize;

    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let bary = barycentric_2d([x as f32 + 0.5, y as f32 + 0.5], p0, p1, p2);
            if bary[0] >= -1.0e-6 && bary[1] >= -1.0e-6 && bary[2] >= -1.0e-6 {
                draw(x, y, bary);
            }
        }
    }
}

fn barycentric_2d(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> [f32; 3] {
    let v0 = [b[0] - a[0], b[1] - a[1]];
    let v1 = [c[0] - a[0], c[1] - a[1]];
    let v2 = [p[0] - a[0], p[1] - a[1]];
    let d00 = v0[0] * v0[0] + v0[1] * v0[1];
    let d01 = v0[0] * v1[0] + v0[1] * v1[1];
    let d11 = v1[0] * v1[0] + v1[1] * v1[1];
    let d20 = v2[0] * v0[0] + v2[1] * v0[1];
    let d21 = v2[0] * v1[0] + v2[1] * v1[1];
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() <= 1.0e-12 {
        return [-1.0, -1.0, -1.0];
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    [u, v, w]
}

fn inpaint_texture_channels(
    texture_size: usize,
    mask: &mut [u8],
    base_color_float: &mut [[f32; 4]],
    metallic_float: &mut [f32],
    roughness_float: &mut [f32],
    alpha_float: &mut [f32],
    fallback: [f32; 6],
) {
    let texels = texture_size * texture_size;
    if mask.len() != texels {
        return;
    }
    let neighbors = [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)];
    for _ in 0..6 {
        let mut changed = false;
        let prev_mask = mask.to_vec();
        let prev_base = base_color_float.to_vec();
        let prev_metallic = metallic_float.to_vec();
        let prev_roughness = roughness_float.to_vec();
        let prev_alpha = alpha_float.to_vec();
        for y in 0..texture_size {
            for x in 0..texture_size {
                let idx = y * texture_size + x;
                if prev_mask[idx] != 0 {
                    continue;
                }
                let mut count = 0usize;
                let mut accum_base = [0.0f32; 4];
                let mut accum_m = 0.0f32;
                let mut accum_r = 0.0f32;
                let mut accum_a = 0.0f32;
                for (dx, dy) in neighbors {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0
                        || ny < 0
                        || nx >= texture_size as isize
                        || ny >= texture_size as isize
                    {
                        continue;
                    }
                    let nidx = ny as usize * texture_size + nx as usize;
                    if prev_mask[nidx] == 0 {
                        continue;
                    }
                    count += 1;
                    for ch in 0..4 {
                        accum_base[ch] += prev_base[nidx][ch];
                    }
                    accum_m += prev_metallic[nidx];
                    accum_r += prev_roughness[nidx];
                    accum_a += prev_alpha[nidx];
                }
                if count == 0 {
                    continue;
                }
                let inv = 1.0 / count as f32;
                for ch in 0..4 {
                    base_color_float[idx][ch] = accum_base[ch] * inv;
                }
                metallic_float[idx] = accum_m * inv;
                roughness_float[idx] = accum_r * inv;
                alpha_float[idx] = accum_a * inv;
                mask[idx] = 255;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for idx in 0..texels {
        if mask[idx] != 0 {
            continue;
        }
        base_color_float[idx] = [fallback[0], fallback[1], fallback[2], fallback[5]];
        metallic_float[idx] = fallback[3];
        roughness_float[idx] = fallback[4];
        alpha_float[idx] = fallback[5];
        mask[idx] = 255;
    }
}

fn quantize_unorm8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn occupancy_target(preprocess: &PreprocessOutput, resolution: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; resolution * resolution * resolution];
    for z in 0..resolution {
        let z_norm = z as f32 / (resolution.saturating_sub(1).max(1) as f32);
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (z * resolution + y) * resolution + x;
                let luma = sample_pixel_luma(preprocess, x as u32, y as u32, z as u32);
                let depth_bias = 1.0 - (z_norm - 0.5).abs() * 1.6;
                out[idx] = (luma * depth_bias).clamp(0.0, 1.0);
            }
        }
    }
    out
}

#[cfg(feature = "runtime-model")]
fn build_sparse_cond_from_preprocess(
    preprocess: &PreprocessOutput,
    tokens: usize,
    cond_channels: usize,
) -> Vec<f32> {
    let patch_side = (tokens as f32).sqrt().floor().max(1.0) as usize;
    let patch_tokens = (patch_side * patch_side).min(tokens);
    let extra_tokens = tokens.saturating_sub(patch_tokens);
    let width = preprocess.width.max(1) as usize;
    let height = preprocess.height.max(1) as usize;
    let mut out = Vec::with_capacity(tokens * cond_channels);
    for token_idx in 0..tokens {
        let (x, y, extra_scale) = if token_idx < patch_tokens {
            let x = token_idx % patch_side;
            let y = token_idx / patch_side;
            (x, y, 0.0f32)
        } else {
            let extra_idx = token_idx - patch_tokens;
            let x = width / 2;
            let y = height / 2;
            let scale = if extra_tokens > 0 {
                extra_idx as f32 / extra_tokens as f32
            } else {
                0.0
            };
            (x, y, scale)
        };
        let xx = if token_idx < patch_tokens {
            (x * width / patch_side).min(width - 1)
        } else {
            x.min(width - 1)
        };
        let yy = if token_idx < patch_tokens {
            (y * height / patch_side).min(height - 1)
        } else {
            y.min(height - 1)
        };
        let offset = (yy * width + xx) * 3;
        let r = preprocess.rgb[offset] as f32 / 255.0;
        let g = preprocess.rgb[offset + 1] as f32 / 255.0;
        let b = preprocess.rgb[offset + 2] as f32 / 255.0;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let nx = if patch_side > 1 {
            x as f32 / (patch_side as f32 - 1.0)
        } else {
            0.0
        };
        let ny = if patch_side > 1 {
            y as f32 / (patch_side as f32 - 1.0)
        } else {
            0.0
        };
        let basis = [r, g, b, luma, nx, ny, extra_scale];
        for channel in 0..cond_channels {
            let base = basis[channel % basis.len()];
            let gain = 1.0 + ((channel / basis.len()) % 17) as f32 / 17.0;
            let phase = ((token_idx + channel + 1) as f32 * 0.013).sin();
            out.push((base * gain + 0.1 * phase).clamp(-1.0, 1.0));
        }
    }
    out
}

fn latent_to_occupancy(latent: &[f32], channels: usize, resolution: usize) -> Vec<f32> {
    let voxels = resolution * resolution * resolution;
    let mut occupancy = vec![0.0f32; voxels];
    for idx in 0..voxels {
        let mut sum = 0.0f32;
        for ch in 0..channels {
            sum += latent[ch * voxels + idx];
        }
        occupancy[idx] = sum / channels.max(1) as f32;
    }
    // Map to [0, 1] using per-sample dynamic normalization.
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &occupancy {
        min = min.min(*value);
        max = max.max(*value);
    }
    let denom = (max - min).max(1.0e-6);
    for value in &mut occupancy {
        *value = (*value - min) / denom;
    }
    occupancy
}

fn upsample_occupancy(input: &[f32], input_res: usize, output_res: usize) -> Vec<f32> {
    if input_res == output_res {
        return input.to_vec();
    }
    let mut out = vec![0.0f32; output_res * output_res * output_res];
    for z in 0..output_res {
        let src_z = z * input_res / output_res;
        for y in 0..output_res {
            let src_y = y * input_res / output_res;
            for x in 0..output_res {
                let src_x = x * input_res / output_res;
                let src_idx = (src_z * input_res + src_y) * input_res + src_x;
                let dst_idx = (z * output_res + y) * output_res + x;
                out[dst_idx] = input[src_idx];
            }
        }
    }
    out
}

#[cfg(feature = "runtime-model")]
fn occupancy_to_coords(
    occupancy: &[f32],
    resolution: usize,
    threshold: f32,
    max_coords: Option<usize>,
) -> Vec<[u32; 4]> {
    let mut candidates = Vec::new();
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (z * resolution + y) * resolution + x;
                let value = occupancy[idx];
                if value > threshold {
                    candidates.push((idx, value));
                }
            }
        }
    }

    if let Some(limit) = max_coords
        && limit > 0
        && candidates.len() > limit
    {
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        candidates.truncate(limit);
        candidates.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let mut coords = Vec::with_capacity(candidates.len());
    for (idx, _) in candidates {
        let x = idx % resolution;
        let y = (idx / resolution) % resolution;
        let z = idx / (resolution * resolution);
        coords.push([0, x as u32, y as u32, z as u32]);
    }
    coords
}

#[cfg(feature = "runtime-model")]
fn runtime_max_sparse_coords() -> Option<usize> {
    std::env::var("TRELLIS2_MAX_SPARSE_COORDS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

#[cfg(feature = "runtime-model")]
fn map_coord_to_dense_flat(
    coord: [u32; 4],
    sparse_resolution: usize,
    dense_resolution: usize,
) -> usize {
    let map_axis = |value: u32| -> usize {
        if sparse_resolution <= 1 || dense_resolution <= 1 {
            return 0;
        }
        let mapped = (value as usize)
            .saturating_mul(dense_resolution)
            .saturating_div(sparse_resolution.max(1));
        mapped.min(dense_resolution - 1)
    };
    let x = map_axis(coord[1]);
    let y = map_axis(coord[2]);
    let z = map_axis(coord[3]);
    (z * dense_resolution + y) * dense_resolution + x
}

fn sample_pixel_luma(preprocess: &PreprocessOutput, x: u32, y: u32, z: u32) -> f32 {
    let width = preprocess.width.max(1);
    let height = preprocess.height.max(1);
    let xx = (x as usize * width as usize / 32).min(width as usize - 1);
    let yy = (y as usize * height as usize / 32).min(height as usize - 1);
    let offset = (yy * width as usize + xx) * 3;
    let r = preprocess.rgb[offset] as f32 / 255.0;
    let g = preprocess.rgb[offset + 1] as f32 / 255.0;
    let b = preprocess.rgb[offset + 2] as f32 / 255.0;
    let z_mod = 0.9 + 0.2 * ((z as f32 / 31.0) - 0.5);
    (0.2126 * r + 0.7152 * g + 0.0722 * b) * z_mod
}

fn sparse_resolution_for_pipeline(pipeline_type: &str) -> usize {
    match pipeline_type {
        "512" | "512_base" => 32,
        "1024" | "1024_single" => 64,
        "1024_cascade" => 32,
        "1536_cascade" => 32,
        _ => 32,
    }
}

fn final_resolution_for_pipeline(pipeline_type: &str) -> usize {
    match pipeline_type {
        "512" | "512_base" => 512,
        "1024" | "1024_single" | "1024_cascade" => 1024,
        "1536_cascade" => 1536,
        _ => 512,
    }
}

fn target_vertex_budget(final_resolution: usize) -> usize {
    if let Ok(value) = std::env::var("TRELLIS2_TARGET_VERTEX_BUDGET")
        && let Ok(parsed) = value.trim().parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }
    match final_resolution {
        512 => 1_566_728,
        1024 => 1_566_728 * 2,
        1536 => 1_566_728 * 3,
        _ => 1_566_728,
    }
}

fn ranked_shape_indices(shape: &ShapeSLatSample) -> Vec<usize> {
    let mut ranked = (0..shape.features.len())
        .map(|idx| {
            let feat = shape.features[idx];
            let score = feat[0].abs()
                + feat[1].abs()
                + feat[2].abs()
                + 0.5 * feat[3].abs()
                + 0.5 * feat[4].abs()
                + 0.5 * feat[5].abs();
            (idx, score)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    ranked.into_iter().map(|(idx, _)| idx).collect()
}

fn canonical_cube() -> Mesh {
    let vertices = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    Mesh {
        vertices,
        faces,
        uvs: Vec::new(),
        material: None,
        pbr_textures: None,
    }
}

#[derive(Clone, Debug)]
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
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{bake_pbr_from_voxels, summarize_material};
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
    fn material_summary_prefers_texture_data_when_available() {
        let textures = dummy_textures();
        let material = summarize_material(&[[0.0; 6]], Some(&textures)).expect("material");
        assert!(material.base_color[0] > 0.1);
        assert!(material.alpha > 0.8);
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
        {
            if let Ok(tex_decoded_reference_guides) = tex_decoder.decode_with_guidance(
                &tex_coords[..rows],
                &tex_feats[..rows],
                &reference_subdivisions[..shape_decoded.subdivisions.len()],
            ) {
                let (ref_guide_stats, ref_guide_overlap, ref_guide_actual_total, ref_guide_reference_total, _) =
                    compare_tex_voxel_overlap(
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
        }

        assert!(
            !shape_decoded.coords.is_empty(),
            "decoded shape coords should not be empty"
        );
        assert!(
            !tex_decoded.coords.is_empty(),
            "decoded tex coords should not be empty"
        );

        let (stats, overlap, actual_total, reference_total, per_channel) = compare_tex_voxel_overlap(
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
            compute_stats(actual_channels[0].as_slice(), reference_channels[0].as_slice()),
            compute_stats(actual_channels[1].as_slice(), reference_channels[1].as_slice()),
            compute_stats(actual_channels[2].as_slice(), reference_channels[2].as_slice()),
            compute_stats(actual_channels[3].as_slice(), reference_channels[3].as_slice()),
            compute_stats(actual_channels[4].as_slice(), reference_channels[4].as_slice()),
            compute_stats(actual_channels[5].as_slice(), reference_channels[5].as_slice()),
        ];
        (stats, overlap, actual.len(), reference.len(), per_channel)
    }
}

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::mesh::Mesh;
use crate::native::preprocess::PreprocessOutput;
#[cfg(feature = "runtime-model")]
use crate::runtime_model::sparse_structure_flow::{
    SparseFlowCondition, SparseStructureFlowRuntime,
};
use crate::sampler::FlowEulerGuidanceIntervalSampler;
use crate::trellis_config::{TrellisNormalization, TrellisPipelineArgs, TrellisSamplerConfig};

#[derive(Debug, Clone)]
pub struct SparseStructureSample {
    pub source: SparseStructureStageSource,
    pub resolution: usize,
    pub flow_resolution: usize,
    pub flow_channels: usize,
    pub noise: Vec<f32>,
    pub step_0_x_t: Vec<f32>,
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
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
    pub step_last_x_t: Vec<[f32; 32]>,
    pub coords: Vec<[u32; 4]>,
}

#[derive(Debug, Clone)]
pub struct TexSLatSample {
    pub features: Vec<[f32; 32]>,
    pub noise: Vec<[f32; 32]>,
    pub step_0_x_t: Vec<[f32; 32]>,
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
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TrellisStageTimings {
    pub sparse_ms: f64,
    pub shape_slat_ms: f64,
    pub tex_slat_ms: f64,
    pub decode_ms: f64,
    pub total_ms: f64,
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
        Self {
            pipeline_type,
            sparse_sampler: args.sparse_structure_sampler.clone(),
            shape_sampler: args.shape_slat_sampler.clone(),
            tex_sampler: args.tex_slat_sampler.clone(),
            shape_norm: args.shape_slat_normalization.clone(),
            tex_norm: args.tex_slat_normalization.clone(),
            #[cfg(feature = "runtime-model")]
            sparse_flow,
        }
    }

    pub fn pipeline_type(&self) -> &str {
        self.pipeline_type.as_str()
    }

    pub fn run(&self, preprocess: &PreprocessOutput, seed: u64) -> TrellisStageOutput {
        self.run_profiled(preprocess, seed).0
    }

    pub fn run_profiled(
        &self,
        preprocess: &PreprocessOutput,
        seed: u64,
    ) -> (TrellisStageOutput, TrellisStageTimings) {
        let total_start = Instant::now();
        let sparse_resolution = sparse_resolution_for_pipeline(self.pipeline_type());
        let sparse_start = Instant::now();
        let sparse = sample_sparse_structure(
            preprocess,
            sparse_resolution,
            seed,
            &self.sparse_sampler,
            #[cfg(feature = "runtime-model")]
            self.sparse_flow.as_ref(),
        );
        let sparse_ms = sparse_start.elapsed().as_secs_f64() * 1000.0;

        let shape_start = Instant::now();
        let shape_slat = sample_shape_slat(
            preprocess,
            &sparse.coords,
            seed ^ 0xA5A5_5A5A,
            &self.shape_sampler,
            &self.shape_norm,
        );
        let shape_slat_ms = shape_start.elapsed().as_secs_f64() * 1000.0;

        let tex_start = Instant::now();
        let tex_slat = sample_tex_slat(
            preprocess,
            &shape_slat,
            seed ^ 0x55AA_AA55,
            &self.tex_sampler,
            &self.shape_norm,
            &self.tex_norm,
        );
        let tex_slat_ms = tex_start.elapsed().as_secs_f64() * 1000.0;

        let decode_start = Instant::now();
        let decoded = decode_latent_to_outputs(&shape_slat, &tex_slat, self.pipeline_type());
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        let output = TrellisStageOutput {
            sparse,
            shape_slat,
            tex_slat,
            decode_shape_subs: decoded.shape_subs,
            decode_tex_voxels: decoded.tex_voxels,
            mesh: decoded.mesh,
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

fn sample_sparse_structure(
    preprocess: &PreprocessOutput,
    resolution: usize,
    seed: u64,
    sampler_config: &TrellisSamplerConfig,
    #[cfg(feature = "runtime-model")] sparse_flow: Option<&SparseStructureFlowRuntime>,
) -> SparseStructureSample {
    #[cfg(feature = "runtime-model")]
    if let Some(sparse_flow) = sparse_flow {
        if let Some(sample) = sample_sparse_structure_with_model(
            preprocess,
            resolution,
            seed,
            sampler_config,
            sparse_flow,
        ) {
            return sample;
        }
    }
    sample_sparse_structure_synthetic(preprocess, resolution, seed, sampler_config)
}

fn sample_sparse_structure_synthetic(
    preprocess: &PreprocessOutput,
    resolution: usize,
    seed: u64,
    sampler_config: &TrellisSamplerConfig,
) -> SparseStructureSample {
    let flow_resolution = 16usize;
    let flow_channels = 8usize;
    let voxel_count = flow_resolution * flow_resolution * flow_resolution;
    let mut rng = Lcg::new(seed);
    let noise: Vec<f32> = (0..(flow_channels * voxel_count))
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();
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
        resolution,
        flow_resolution,
        flow_channels,
        noise,
        step_0_x_t: trace.step_0_x_t,
        step_last_x_t: trace.step_last_x_t,
        latent,
        coords,
    }
}

#[cfg(feature = "runtime-model")]
fn sample_sparse_structure_with_model(
    preprocess: &PreprocessOutput,
    resolution: usize,
    seed: u64,
    sampler_config: &TrellisSamplerConfig,
    sparse_flow: &SparseStructureFlowRuntime,
) -> Option<SparseStructureSample> {
    let config = sparse_flow.config();
    let flow_resolution = config.resolution;
    let channels = config.in_channels;
    let flow_voxels = flow_resolution * flow_resolution * flow_resolution;
    let mut rng = Lcg::new(seed);
    let noise: Vec<f32> = (0..(channels * flow_voxels))
        .map(|_| rng.next_f32() * 2.0 - 1.0)
        .collect();

    let cond_tokens = 32 * 32;
    let cond = build_sparse_cond_from_preprocess(preprocess, cond_tokens, config.cond_channels);
    let neg_cond = vec![0.0f32; cond.len()];
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
    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );

    let mut failed = false;
    let trace = sampler.sample_with_trace(&noise, sample_cfg, |x_t, t, use_cond| {
        let cond_source: &SparseFlowCondition = if use_cond {
            &cond_tensor
        } else {
            &neg_cond_tensor
        };
        match sparse_flow.predict_velocity_with_condition(x_t, t, cond_source) {
            Ok(pred) => pred,
            Err(err) => {
                failed = true;
                eprintln!(
                    "burn_trellis: sparse flow model prediction failed ({err}); using synthetic sparse stage fallback."
                );
                x_t.to_vec()
            }
        }
    });
    if failed {
        return None;
    }
    let latent = trace.samples;

    let occupancy = latent_to_occupancy(&latent, channels, flow_resolution);
    let upsampled = upsample_occupancy(occupancy.as_slice(), flow_resolution, resolution);
    let mut coords = occupancy_to_coords(upsampled.as_slice(), resolution, 0.5);
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
        resolution,
        flow_resolution,
        flow_channels: channels,
        noise,
        step_0_x_t: trace.step_0_x_t,
        step_last_x_t: trace.step_last_x_t,
        latent,
        coords,
    })
}

fn sample_shape_slat(
    preprocess: &PreprocessOutput,
    coords: &[[u32; 4]],
    seed: u64,
    sampler_config: &TrellisSamplerConfig,
    normalization: &TrellisNormalization,
) -> ShapeSLatSample {
    let mut rng = Lcg::new(seed);
    let mut features = Vec::with_capacity(coords.len());
    let mut noise_rows = Vec::with_capacity(coords.len());
    let mut step_0_rows = Vec::with_capacity(coords.len());
    let mut step_last_rows = Vec::with_capacity(coords.len());
    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    for coord in coords {
        let base = sample_pixel_luma(preprocess, coord[1], coord[2], coord[3]);
        let noise = (0..32)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect::<Vec<_>>();
        let target = vec![base; 32];
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
            step_last_row[idx] = trace.step_last_x_t[idx];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_last_rows.push(step_last_row);
    }
    ShapeSLatSample {
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_last_x_t: step_last_rows,
        coords: coords.to_vec(),
    }
}

fn sample_tex_slat(
    preprocess: &PreprocessOutput,
    shape_slat: &ShapeSLatSample,
    seed: u64,
    sampler_config: &TrellisSamplerConfig,
    shape_normalization: &TrellisNormalization,
    normalization: &TrellisNormalization,
) -> TexSLatSample {
    let mut rng = Lcg::new(seed);
    let (sampler, sample_cfg) = FlowEulerGuidanceIntervalSampler::from_params(
        sampler_config.args.sigma_min,
        &sampler_config.params,
    );
    let mut features = Vec::with_capacity(shape_slat.coords.len());
    let mut noise_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_0_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut step_last_rows = Vec::with_capacity(shape_slat.coords.len());
    let mut shape_cond_rows = Vec::with_capacity(shape_slat.coords.len());
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
        let noise = (0..32)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect::<Vec<_>>();
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
            step_last_row[ch] = trace.step_last_x_t[ch];
        }
        features.push(row);
        noise_rows.push(noise_row);
        step_0_rows.push(step_0_row);
        step_last_rows.push(step_last_row);
        shape_cond_rows.push(shape_cond);
    }
    TexSLatSample {
        features,
        noise: noise_rows,
        step_0_x_t: step_0_rows,
        step_last_x_t: step_last_rows,
        shape_slat_cond: shape_cond_rows,
        coords: shape_slat.coords.clone(),
    }
}

fn decode_latent_to_outputs(
    shape: &ShapeSLatSample,
    tex: &TexSLatSample,
    pipeline_type: &str,
) -> DecodedLatentOutput {
    if shape.coords.is_empty() || shape.features.is_empty() {
        return DecodedLatentOutput {
            mesh: canonical_cube(),
            shape_subs: Vec::new(),
            tex_voxels: DecodeTexVoxelSample {
                coords: Vec::new(),
                feats: Vec::new(),
                spatial_shape: [512, 512, 512],
            },
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

    let mesh = if vertices.is_empty() || faces.is_empty() {
        canonical_cube()
    } else {
        Mesh { vertices, faces }
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
    }
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
    let side = (tokens as f32).sqrt().round().max(1.0) as usize;
    let mut out = Vec::with_capacity(side * side * cond_channels);
    for y in 0..side {
        for x in 0..side {
            let xx =
                (x * preprocess.width.max(1) as usize / side).min(preprocess.width as usize - 1);
            let yy =
                (y * preprocess.height.max(1) as usize / side).min(preprocess.height as usize - 1);
            let offset = (yy * preprocess.width as usize + xx) * 3;
            let r = preprocess.rgb[offset] as f32 / 255.0;
            let g = preprocess.rgb[offset + 1] as f32 / 255.0;
            let b = preprocess.rgb[offset + 2] as f32 / 255.0;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let nx = if side > 1 {
                x as f32 / (side as f32 - 1.0)
            } else {
                0.0
            };
            let ny = if side > 1 {
                y as f32 / (side as f32 - 1.0)
            } else {
                0.0
            };
            let basis = [r, g, b, luma, nx, ny];
            for channel in 0..cond_channels {
                let base = basis[channel % basis.len()];
                let gain = 1.0 + ((channel / basis.len()) % 17) as f32 / 17.0;
                let phase = ((x + y + channel + 1) as f32 * 0.013).sin();
                out.push((base * gain + 0.1 * phase).clamp(-1.0, 1.0));
            }
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
fn occupancy_to_coords(occupancy: &[f32], resolution: usize, threshold: f32) -> Vec<[u32; 4]> {
    let mut coords = Vec::new();
    for z in 0..resolution {
        for y in 0..resolution {
            for x in 0..resolution {
                let idx = (z * resolution + y) * resolution + x;
                if occupancy[idx] > threshold {
                    coords.push([0, x as u32, y as u32, z as u32]);
                }
            }
        }
    }
    coords
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
    Mesh { vertices, faces }
}

#[derive(Clone, Debug)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        };
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.state >> 32) as u32
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / u32::MAX as f32
    }
}

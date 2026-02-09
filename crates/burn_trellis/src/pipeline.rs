use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::ValueEnum;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::TrellisQuality;
use crate::hook_diff::{HookSnapshot, HookTensor};
use crate::hook_trace::HookTrace;
use crate::mesh::{Mesh, write_obj_mesh};
use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
use crate::preprocess::{
    PreprocessConfig, PreprocessOutput, preprocess_image, preprocess_image_path,
};
use crate::sampler::{FlowEulerSampleConfig, timestep_pairs};
use crate::staged_pipeline::{
    SamplerConfigOverride, SparseRowNoiseOverride, SparseStructureStageSource,
    TrellisNoiseOverrides, TrellisStageOutput, TrellisStageRuntime,
};
use crate::trellis_config::TrellisPipelineConfig;

const HOOK_MAX_ROWS: usize = usize::MAX;
const HOOK_MAX_DENSE_ELEMENTS: usize = usize::MAX;
const HOOK_SAMPLER_SNAPSHOTS: usize = 3;

#[cfg(feature = "runtime-model")]
fn reset_runtime_transfer_stats() {
    crate::runtime_model::sparse_structure_flow::reset_host_transfer_stats();
}

#[cfg(not(feature = "runtime-model"))]
fn reset_runtime_transfer_stats() {}

#[cfg(feature = "runtime-model")]
fn runtime_transfer_stats() -> (u64, u64) {
    let stats = crate::runtime_model::sparse_structure_flow::host_transfer_stats();
    (stats.readback_count, stats.readback_elements)
}

#[cfg(not(feature = "runtime-model"))]
fn runtime_transfer_stats() -> (u64, u64) {
    (0, 0)
}

fn hook_full_capture_enabled() -> bool {
    true
}

fn hook_max_rows() -> usize {
    HOOK_MAX_ROWS
}

fn hook_max_dense_elements() -> usize {
    HOOK_MAX_DENSE_ELEMENTS
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrellisDevice {
    #[default]
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

impl TrellisDevice {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TrellisRunOptions {
    pub quality: TrellisQuality,
    pub device: TrellisDevice,
    pub seed: Option<u64>,
    pub hook_output: Option<PathBuf>,
    pub noise_overrides_hook: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct Trellis2PipelineConfig {
    pub weights_root: PathBuf,
    pub image_large_root: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrellisPipelineTimings {
    pub preprocess_ms: f64,
    pub runtime_setup_ms: f64,
    pub sparse_ms: f64,
    pub shape_slat_ms: f64,
    pub tex_slat_ms: f64,
    pub decode_ms: f64,
    pub hook_capture_ms: f64,
    pub host_readback_count: u64,
    pub host_readback_elements: u64,
    pub total_ms: f64,
}

#[derive(Clone, Debug)]
pub struct TrellisInferenceProfile {
    pub mesh: Mesh,
    pub timings: TrellisPipelineTimings,
    pub sparse_source: SparseStructureStageSource,
}

impl Default for Trellis2PipelineConfig {
    fn default() -> Self {
        let image_large_root = resolve_trellis2_image_large_root(None);
        Self {
            weights_root: resolve_trellis2_weights_root(None),
            image_large_root: if image_large_root.exists() {
                Some(image_large_root)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrellisRuntimeError {
    message: String,
}

impl TrellisRuntimeError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for TrellisRuntimeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for TrellisRuntimeError {}

pub struct Trellis2Pipeline {
    config: Trellis2PipelineConfig,
}

impl Trellis2Pipeline {
    pub fn from_pretrained(weights_root: impl AsRef<Path>) -> Result<Self, TrellisRuntimeError> {
        let weights_root = resolve_trellis2_weights_root(Some(weights_root.as_ref()));
        let config = Trellis2PipelineConfig {
            weights_root,
            ..Default::default()
        };
        Self::new(config)
    }

    pub fn new(config: Trellis2PipelineConfig) -> Result<Self, TrellisRuntimeError> {
        #[cfg(not(target_arch = "wasm32"))]
        if !config.weights_root.exists() {
            return Err(TrellisRuntimeError::new(format!(
                "Trellis2 weights root does not exist: {}",
                config.weights_root.display()
            )));
        }
        Ok(Self { config })
    }

    pub fn config(&self) -> &Trellis2PipelineConfig {
        &self.config
    }

    pub fn validate_runtime(&self) -> Result<(), TrellisRuntimeError> {
        let pipeline_path = self.config.weights_root.join("pipeline.json");
        if !pipeline_path.exists() {
            return Err(TrellisRuntimeError::new(format!(
                "missing Trellis2 pipeline.json: {}",
                pipeline_path.display()
            )));
        }

        let pipeline_bytes = std::fs::read(&pipeline_path).map_err(|err| {
            TrellisRuntimeError::new(format!(
                "failed to read Trellis2 pipeline config '{}': {err}",
                pipeline_path.display()
            ))
        })?;
        let pipeline_json: Value = serde_json::from_slice(&pipeline_bytes).map_err(|err| {
            TrellisRuntimeError::new(format!(
                "failed to parse Trellis2 pipeline config '{}': {err}",
                pipeline_path.display()
            ))
        })?;
        let model_stems = collect_model_stems(&pipeline_json);
        if model_stems.is_empty() {
            return Err(TrellisRuntimeError::new(format!(
                "Trellis2 pipeline config '{}' has no model stems in args.models",
                pipeline_path.display()
            )));
        }

        let mut missing = Vec::new();
        for stem in model_stems {
            let config_path = resolve_model_source_path(
                &stem,
                "json",
                &self.config.weights_root,
                self.config.image_large_root.as_deref(),
            );
            if !config_path.exists() {
                missing.push(config_path.display().to_string());
            }

            let safetensors_path = resolve_model_source_path(
                &stem,
                "safetensors",
                &self.config.weights_root,
                self.config.image_large_root.as_deref(),
            );
            let bpk_path = safetensors_path.with_extension("bpk");
            let bpk_f16_path = with_file_stem_suffix(&bpk_path, "_f16");
            if !safetensors_path.exists() && !bpk_path.exists() && !bpk_f16_path.exists() {
                missing.push(format!(
                    "{} (or {} / {})",
                    safetensors_path.display(),
                    bpk_path.display(),
                    bpk_f16_path.display()
                ));
            }
        }

        if missing.is_empty() {
            Ok(())
        } else {
            let preview = missing
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            let suffix = if missing.len() > 8 {
                format!("\n... and {} more", missing.len() - 8)
            } else {
                String::new()
            };
            Err(TrellisRuntimeError::new(format!(
                "Trellis2 runtime assets are incomplete ({} missing):\n{}{}",
                missing.len(),
                preview,
                suffix
            )))
        }
    }

    pub fn infer_mesh(
        &self,
        image_path: &Path,
        options: &TrellisRunOptions,
    ) -> Result<Mesh, TrellisRuntimeError> {
        let profiled = self.infer_mesh_profile(image_path, options)?;
        Ok(profiled.mesh)
    }

    pub fn infer_mesh_profile(
        &self,
        image_path: &Path,
        options: &TrellisRunOptions,
    ) -> Result<TrellisInferenceProfile, TrellisRuntimeError> {
        let total_start = Instant::now();
        let preprocess_start = Instant::now();
        let preprocess = preprocess_image_path(image_path, PreprocessConfig::default())
            .map_err(|err| TrellisRuntimeError::new(format!("preprocess failed: {err}")))?;
        let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
        let setup_start = Instant::now();
        let runtime = self.load_stage_runtime(options)?;
        let runtime_setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;
        let seed = options.seed.unwrap_or(42);
        let noise_overrides = self.load_noise_overrides(options)?;
        reset_runtime_transfer_stats();
        let (stage_output, stage_timings) = runtime.run_profiled_with_overrides(
            &preprocess,
            seed,
            noise_overrides.as_ref(),
            options.hook_output.is_some(),
        );
        let (host_readback_count, host_readback_elements) = runtime_transfer_stats();
        let hook_capture_start = Instant::now();
        self.capture_pipeline_hook(&preprocess, &stage_output, runtime.pipeline_type(), options)?;
        let hook_capture_ms = hook_capture_start.elapsed().as_secs_f64() * 1000.0;
        let timings = TrellisPipelineTimings {
            preprocess_ms,
            runtime_setup_ms,
            sparse_ms: stage_timings.sparse_ms,
            shape_slat_ms: stage_timings.shape_slat_ms,
            tex_slat_ms: stage_timings.tex_slat_ms,
            decode_ms: stage_timings.decode_ms,
            hook_capture_ms,
            host_readback_count,
            host_readback_elements,
            total_ms: total_start.elapsed().as_secs_f64() * 1000.0,
        };
        Ok(TrellisInferenceProfile {
            sparse_source: stage_output.sparse.source,
            mesh: stage_output.mesh,
            timings,
        })
    }

    pub fn infer_mesh_from_image_bytes(
        &self,
        image_bytes: &[u8],
        options: &TrellisRunOptions,
    ) -> Result<Mesh, TrellisRuntimeError> {
        let image = image::load_from_memory(image_bytes).map_err(|err| {
            TrellisRuntimeError::new(format!("failed to decode input image bytes: {err}"))
        })?;
        self.infer_mesh_from_image(image, options)
    }

    pub fn infer_mesh_from_image(
        &self,
        image: DynamicImage,
        options: &TrellisRunOptions,
    ) -> Result<Mesh, TrellisRuntimeError> {
        let preprocess = preprocess_image(image, PreprocessConfig::default())
            .map_err(|err| TrellisRuntimeError::new(format!("preprocess failed: {err}")))?;
        let runtime = self.load_stage_runtime(options)?;
        let seed = options.seed.unwrap_or(42);
        let noise_overrides = self.load_noise_overrides(options)?;
        reset_runtime_transfer_stats();
        let stage_output = runtime
            .run_profiled_with_overrides(
                &preprocess,
                seed,
                noise_overrides.as_ref(),
                options.hook_output.is_some(),
            )
            .0;
        self.capture_pipeline_hook(&preprocess, &stage_output, runtime.pipeline_type(), options)?;
        Ok(stage_output.mesh)
    }

    pub fn infer_mesh_to_obj(
        &self,
        image_path: &Path,
        output_obj: &Path,
        options: &TrellisRunOptions,
    ) -> Result<(), TrellisRuntimeError> {
        let mesh = self.infer_mesh(image_path, options)?;
        write_obj_mesh(output_obj, &mesh)
            .map_err(|err| TrellisRuntimeError::new(format!("failed to write OBJ: {err}")))
    }

    fn load_stage_runtime(
        &self,
        options: &TrellisRunOptions,
    ) -> Result<TrellisStageRuntime, TrellisRuntimeError> {
        let pipeline_path = self.config.weights_root.join("pipeline.json");
        let pipeline_bytes = std::fs::read(&pipeline_path).map_err(|err| {
            TrellisRuntimeError::new(format!(
                "failed to read Trellis2 pipeline config '{}': {err}",
                pipeline_path.display()
            ))
        })?;
        let pipeline = TrellisPipelineConfig::from_json_bytes(&pipeline_bytes).map_err(|err| {
            TrellisRuntimeError::new(format!(
                "failed to parse Trellis2 pipeline config '{}': {err}",
                pipeline_path.display()
            ))
        })?;
        let preferred_pipeline_type = options.quality.settings().pipeline_type;
        let prefer_wgpu = !matches!(options.device, TrellisDevice::Cpu);
        Ok(TrellisStageRuntime::from_args_with_assets(
            &pipeline.args,
            Some(preferred_pipeline_type),
            Some(self.config.weights_root.as_path()),
            self.config.image_large_root.as_deref(),
            prefer_wgpu,
        ))
    }

    fn load_noise_overrides(
        &self,
        options: &TrellisRunOptions,
    ) -> Result<Option<TrellisNoiseOverrides>, TrellisRuntimeError> {
        let Some(path) = options.noise_overrides_hook.as_ref() else {
            return Ok(None);
        };
        let snapshot = HookSnapshot::from_file(path).map_err(|err| {
            TrellisRuntimeError::new(format!(
                "failed to load noise override hook '{}': {err}",
                path.display()
            ))
        })?;

        let mut overrides = TrellisNoiseOverrides::default();
        if let Some(tensor) = snapshot.tensors.get("sample_sparse_structure.noise") {
            overrides.sparse_noise = Some(tensor.data.clone());
        }
        overrides.shape_noise = extract_sparse_row_noise_override(
            &snapshot,
            "sample_shape_slat.noise",
            path.as_path(),
        )?;
        overrides.tex_noise =
            extract_sparse_row_noise_override(&snapshot, "sample_tex_slat.noise", path.as_path())?;
        overrides.shape_noise_dense =
            extract_dense_f32_override(&snapshot, "sample_shape_slat.noise_dense");
        overrides.tex_noise_dense =
            extract_dense_f32_override(&snapshot, "sample_tex_slat.noise_dense");
        overrides.sparse_sampler = extract_sampler_override(
            &snapshot,
            "sample_sparse_structure.sampler.config",
            path.as_path(),
        )?;
        overrides.shape_sampler = extract_sampler_override(
            &snapshot,
            "sample_shape_slat.sampler.config",
            path.as_path(),
        )?;
        overrides.tex_sampler =
            extract_sampler_override(&snapshot, "sample_tex_slat.sampler.config", path.as_path())?;
        overrides.cond_512 = extract_dense_f32_override(&snapshot, "get_cond_512.out.cond");
        overrides.neg_cond_512 = extract_dense_f32_override(&snapshot, "get_cond_512.out.neg_cond");
        overrides.cond_1024 = extract_dense_f32_override(&snapshot, "get_cond_1024.out.cond");
        overrides.neg_cond_1024 =
            extract_dense_f32_override(&snapshot, "get_cond_1024.out.neg_cond");

        if overrides.is_empty() {
            return Err(TrellisRuntimeError::new(format!(
                "noise override hook '{}' does not contain any supported noise tensors",
                path.display()
            )));
        }
        eprintln!(
            "burn_trellis: loaded noise overrides from '{}': sparse={} shape_rows={} tex_rows={} shape_dense={} tex_dense={} sparse_sampler={} shape_sampler={} tex_sampler={} cond512={} neg512={} cond1024={} neg1024={}",
            path.display(),
            overrides.sparse_noise.as_ref().map_or(0usize, |v| v.len()),
            overrides
                .shape_noise
                .as_ref()
                .map_or(0usize, |v| v.coords.len()),
            overrides
                .tex_noise
                .as_ref()
                .map_or(0usize, |v| v.coords.len()),
            overrides
                .shape_noise_dense
                .as_ref()
                .map_or(0usize, |v| v.len()),
            overrides
                .tex_noise_dense
                .as_ref()
                .map_or(0usize, |v| v.len()),
            overrides.sparse_sampler.is_some(),
            overrides.shape_sampler.is_some(),
            overrides.tex_sampler.is_some(),
            overrides.cond_512.as_ref().map_or(0usize, |v| v.len()),
            overrides.neg_cond_512.as_ref().map_or(0usize, |v| v.len()),
            overrides.cond_1024.as_ref().map_or(0usize, |v| v.len()),
            overrides.neg_cond_1024.as_ref().map_or(0usize, |v| v.len()),
        );
        Ok(Some(overrides))
    }

    fn capture_pipeline_hook(
        &self,
        preprocess: &PreprocessOutput,
        stage_output: &TrellisStageOutput,
        pipeline_type: &str,
        options: &TrellisRunOptions,
    ) -> Result<(), TrellisRuntimeError> {
        if let Some(hook_output) = options.hook_output.as_ref() {
            let mut trace = HookTrace::default();
            let preprocess_shape = vec![
                preprocess.height as usize,
                preprocess.width as usize,
                3usize,
            ];
            let (preprocess_hook_shape, preprocess_hook_rgb) =
                sample_dense_u8_for_hook(preprocess_shape.as_slice(), preprocess.rgb.as_slice());
            trace
                .insert_u8(
                    "preprocess_image.output",
                    preprocess_hook_shape.clone(),
                    preprocess_hook_rgb.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_u8("run.image", preprocess_hook_shape, preprocess_hook_rgb)
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "run.final_resolution",
                    vec![1],
                    vec![final_resolution_for_pipeline(pipeline_type) as f32],
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "run.sparse_structure_resolution",
                    vec![1],
                    vec![stage_output.sparse.resolution as f32],
                )
                .map_err(TrellisRuntimeError::new)?;

            let cond_channels = 1024usize;
            let cond_512_tokens = 32usize * 32usize + 5usize;
            let cond = build_synthetic_cond_trace(preprocess, cond_512_tokens, cond_channels);
            let (cond_shape, cond_values) = sample_dense_f32_for_hook(
                &[1usize, cond_512_tokens, cond_channels],
                cond.as_slice(),
            );
            trace
                .insert_f32("get_cond_512.out.cond", cond_shape.clone(), cond_values)
                .map_err(TrellisRuntimeError::new)?;
            let neg_cond = vec![0.0f32; cond_512_tokens * cond_channels];
            let (neg_cond_shape, neg_cond_values) = sample_dense_f32_for_hook(
                &[1usize, cond_512_tokens, cond_channels],
                neg_cond.as_slice(),
            );
            trace
                .insert_f32("get_cond_512.out.neg_cond", neg_cond_shape, neg_cond_values)
                .map_err(TrellisRuntimeError::new)?;
            if pipeline_type != "512" && pipeline_type != "512_base" {
                let cond_1024_tokens = 64usize * 64usize + 5usize;
                let cond_1024 =
                    build_synthetic_cond_trace(preprocess, cond_1024_tokens, cond_channels);
                let (cond_1024_shape, cond_1024_values) = sample_dense_f32_for_hook(
                    &[1usize, cond_1024_tokens, cond_channels],
                    cond_1024.as_slice(),
                );
                trace
                    .insert_f32(
                        "get_cond_1024.out.cond",
                        cond_1024_shape.clone(),
                        cond_1024_values,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                let neg_cond_1024 = vec![0.0f32; cond_1024_tokens * cond_channels];
                let (neg_cond_1024_shape, neg_cond_1024_values) = sample_dense_f32_for_hook(
                    &[1usize, cond_1024_tokens, cond_channels],
                    neg_cond_1024.as_slice(),
                );
                trace
                    .insert_f32(
                        "get_cond_1024.out.neg_cond",
                        neg_cond_1024_shape,
                        neg_cond_1024_values,
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }

            let sparse_shape = vec![
                1usize,
                stage_output.sparse.flow_channels,
                stage_output.sparse.flow_resolution,
                stage_output.sparse.flow_resolution,
                stage_output.sparse.flow_resolution,
            ];
            trace
                .insert_f32(
                    "sample_sparse_structure.noise",
                    sparse_shape.clone(),
                    stage_output.sparse.noise.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            insert_sampler_hook_config(
                &mut trace,
                "sample_sparse_structure.sampler",
                stage_output.sparse.sampler_config,
                stage_output.sparse.sigma_min,
            )
            .map_err(TrellisRuntimeError::new)?;
            for step_idx in
                sampler_snapshot_steps(stage_output.sparse.step_count, HOOK_SAMPLER_SNAPSHOTS)
            {
                let key = format!(
                    "sample_sparse_structure.sampler.step_{step_idx:03}_of_{:03}.x_t",
                    stage_output.sparse.step_count.max(1)
                );
                trace
                    .insert_f32(
                        key,
                        sparse_shape.clone(),
                        sparse_step_values(&stage_output.sparse, step_idx),
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }
            trace
                .insert_f32(
                    "sample_sparse_structure.latent",
                    sparse_shape,
                    stage_output.sparse.latent.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            let sparse_indices = sampled_row_indices(stage_output.sparse.coords.len(), 4);
            trace
                .insert_f32(
                    "sample_sparse_structure.coords",
                    vec![sparse_indices.len(), 4],
                    flatten_coords_indices(&stage_output.sparse.coords, sparse_indices.as_slice()),
                )
                .map_err(TrellisRuntimeError::new)?;
            if let Some(dense_noise) = stage_output.shape_slat.dense_noise.as_ref()
                && stage_output.shape_slat.dense_resolution > 0
                && stage_output.shape_slat.dense_channels > 0
            {
                trace
                    .insert_f32(
                        "sample_shape_slat.noise_dense",
                        vec![
                            1,
                            stage_output.shape_slat.dense_channels,
                            stage_output.shape_slat.dense_resolution,
                            stage_output.shape_slat.dense_resolution,
                            stage_output.shape_slat.dense_resolution,
                        ],
                        dense_noise.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }

            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.noise",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.noise,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sampler_hook_config(
                &mut trace,
                "sample_shape_slat.sampler",
                stage_output.shape_slat.sampler_config,
                stage_output.shape_slat.sigma_min,
            )
            .map_err(TrellisRuntimeError::new)?;
            for step_idx in
                sampler_snapshot_steps(stage_output.shape_slat.step_count, HOOK_SAMPLER_SNAPSHOTS)
            {
                let prefix = format!(
                    "sample_shape_slat.sampler.step_{step_idx:03}_of_{:03}.x_t",
                    stage_output.shape_slat.step_count.max(1)
                );
                insert_sparse_trace_rows(
                    &mut trace,
                    prefix.as_str(),
                    &stage_output.shape_slat.coords,
                    shape_slat_step_values(&stage_output.shape_slat, step_idx),
                    32,
                    stage_output.sparse.resolution,
                )
                .map_err(TrellisRuntimeError::new)?;
            }
            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.slat",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.features,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            if let Some(dense_noise) = stage_output.tex_slat.dense_noise.as_ref()
                && stage_output.tex_slat.dense_resolution > 0
                && stage_output.tex_slat.dense_channels > 0
            {
                trace
                    .insert_f32(
                        "sample_tex_slat.noise_dense",
                        vec![
                            1,
                            stage_output.tex_slat.dense_channels,
                            stage_output.tex_slat.dense_resolution,
                            stage_output.tex_slat.dense_resolution,
                            stage_output.tex_slat.dense_resolution,
                        ],
                        dense_noise.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }

            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.noise",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.noise,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sampler_hook_config(
                &mut trace,
                "sample_tex_slat.sampler",
                stage_output.tex_slat.sampler_config,
                stage_output.tex_slat.sigma_min,
            )
            .map_err(TrellisRuntimeError::new)?;
            for step_idx in
                sampler_snapshot_steps(stage_output.tex_slat.step_count, HOOK_SAMPLER_SNAPSHOTS)
            {
                let prefix = format!(
                    "sample_tex_slat.sampler.step_{step_idx:03}_of_{:03}.x_t",
                    stage_output.tex_slat.step_count.max(1)
                );
                insert_sparse_trace_rows(
                    &mut trace,
                    prefix.as_str(),
                    &stage_output.tex_slat.coords,
                    tex_slat_step_values(&stage_output.tex_slat, step_idx),
                    32,
                    stage_output.sparse.resolution,
                )
                .map_err(TrellisRuntimeError::new)?;
            }
            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.shape_slat_cond",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.shape_slat_cond,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.slat",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.features,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;

            let mesh_vertex_indices = sampled_row_indices(stage_output.mesh.vertices.len(), 3);
            let mesh_face_indices = sampled_row_indices(stage_output.mesh.faces.len(), 3);
            trace
                .insert_f32(
                    "decode_shape_slat.meshes.0.vertices",
                    vec![mesh_vertex_indices.len(), 3],
                    flatten_vertices_indices(
                        &stage_output.mesh.vertices,
                        mesh_vertex_indices.as_slice(),
                    ),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_shape_slat.meshes.0.vertices_count",
                    vec![1],
                    vec![stage_output.mesh.vertices.len() as f32],
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_shape_slat.meshes.0.faces",
                    vec![mesh_face_indices.len(), 3],
                    flatten_faces_indices(&stage_output.mesh.faces, mesh_face_indices.as_slice()),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_shape_slat.meshes.0.faces_count",
                    vec![1],
                    vec![stage_output.mesh.faces.len() as f32],
                )
                .map_err(TrellisRuntimeError::new)?;

            for level in 0..4usize {
                let prefix = format!("decode_shape_slat.subs.{level}");
                if let Some(sub) = stage_output.decode_shape_subs.get(level) {
                    let sub_indices = sampled_row_indices(sub.coords.len(), 4);
                    let sub_rows = sub_indices.len();
                    trace
                        .insert_f32(
                            format!("{prefix}.coords"),
                            vec![sub_rows, 4],
                            flatten_coords_indices(&sub.coords, sub_indices.as_slice()),
                        )
                        .map_err(TrellisRuntimeError::new)?;
                    trace
                        .insert_f32(
                            format!("{prefix}.feats"),
                            vec![sub_rows, 8],
                            flatten_feature_rows_indices_fixed(
                                &sub.feats,
                                sub_indices.as_slice(),
                                8,
                            ),
                        )
                        .map_err(TrellisRuntimeError::new)?;
                    trace
                        .insert_f32(format!("{prefix}.shape"), vec![2], vec![1.0, 8.0])
                        .map_err(TrellisRuntimeError::new)?;
                    trace
                        .insert_f32(
                            format!("{prefix}.spatial_shape"),
                            vec![3],
                            vec![
                                sub.spatial_shape[0] as f32,
                                sub.spatial_shape[1] as f32,
                                sub.spatial_shape[2] as f32,
                            ],
                        )
                        .map_err(TrellisRuntimeError::new)?;
                    continue;
                }
                trace
                    .insert_f32(format!("{prefix}.coords"), vec![0, 4], Vec::new())
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(format!("{prefix}.feats"), vec![0, 8], Vec::new())
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(format!("{prefix}.shape"), vec![2], vec![1.0, 8.0])
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        format!("{prefix}.spatial_shape"),
                        vec![3],
                        vec![0.0, 0.0, 0.0],
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }

            let voxel_source_coords = &stage_output.decode_tex_voxels.coords;
            let voxel_indices = sampled_row_indices(voxel_source_coords.len(), 4);
            let voxel_rows = voxel_indices.len();
            let mut voxel_attrs = Vec::with_capacity(voxel_rows * 6);
            for idx in voxel_indices.iter().copied() {
                if let Some(row) = stage_output.decode_tex_voxels.feats.get(idx) {
                    voxel_attrs.extend(row.iter().copied());
                } else {
                    voxel_attrs.extend_from_slice(&[0.0; 6]);
                }
            }
            trace
                .insert_f32(
                    "decode_tex_slat.voxels.coords",
                    vec![voxel_rows, 4],
                    flatten_coords_indices(voxel_source_coords, voxel_indices.as_slice()),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_tex_slat.voxels.feats",
                    vec![voxel_rows, 6],
                    voxel_attrs.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32("decode_tex_slat.voxels.shape", vec![2], vec![1.0, 6.0])
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_tex_slat.voxels.spatial_shape",
                    vec![3],
                    vec![
                        stage_output.decode_tex_voxels.spatial_shape[0] as f32,
                        stage_output.decode_tex_voxels.spatial_shape[1] as f32,
                        stage_output.decode_tex_voxels.spatial_shape[2] as f32,
                    ],
                )
                .map_err(TrellisRuntimeError::new)?;

            trace
                .insert_f32(
                    "decode_latent.mesh.0.vertices",
                    vec![mesh_vertex_indices.len(), 3],
                    flatten_vertices_indices(
                        &stage_output.mesh.vertices,
                        mesh_vertex_indices.as_slice(),
                    ),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.vertices_count",
                    vec![1],
                    vec![stage_output.mesh.vertices.len() as f32],
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.faces",
                    vec![mesh_face_indices.len(), 3],
                    flatten_faces_indices(&stage_output.mesh.faces, mesh_face_indices.as_slice()),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.faces_count",
                    vec![1],
                    vec![stage_output.mesh.faces.len() as f32],
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.voxel_coords",
                    vec![voxel_rows, 3],
                    flatten_coords3_indices(voxel_source_coords, voxel_indices.as_slice()),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.voxel_attrs",
                    vec![voxel_rows, 6],
                    voxel_attrs,
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "decode_latent.mesh.0.voxel_count",
                    vec![1],
                    vec![voxel_source_coords.len() as f32],
                )
                .map_err(TrellisRuntimeError::new)?;

            if let Some(pbr) = stage_output.pbr.as_ref() {
                let uv_vertices = stage_output
                    .mesh
                    .vertices
                    .iter()
                    .flat_map(|value| value.iter().copied())
                    .collect::<Vec<_>>();
                let uv_faces = stage_output
                    .mesh
                    .faces
                    .iter()
                    .flat_map(|value| value.iter().map(|index| *index as f32))
                    .collect::<Vec<_>>();
                let uv_coords = pbr
                    .uvs
                    .iter()
                    .flat_map(|value| value.iter().copied())
                    .collect::<Vec<_>>();
                let sample_positions = pbr
                    .sample_positions
                    .iter()
                    .flat_map(|value| value.iter().copied())
                    .collect::<Vec<_>>();
                let sample_attrs = pbr
                    .sample_attrs
                    .iter()
                    .flat_map(|value| value.iter().copied())
                    .collect::<Vec<_>>();
                let base_color_float = pbr
                    .base_color_float
                    .iter()
                    .flat_map(|value| value.iter().copied())
                    .collect::<Vec<_>>();

                trace
                    .insert_f32(
                        "pbr.uv_unwrap.vertices",
                        vec![stage_output.mesh.vertices.len(), 3],
                        uv_vertices,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.uv_unwrap.faces",
                        vec![stage_output.mesh.faces.len(), 3],
                        uv_faces,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32("pbr.uv_unwrap.uvs", vec![pbr.uvs.len(), 2], uv_coords)
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_u8(
                        "pbr.raster.mask",
                        vec![pbr.texture_height, pbr.texture_width],
                        pbr.raster_mask.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.sample.position",
                        vec![pbr.sample_positions.len(), 3],
                        sample_positions,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.sample.attrs_float",
                        vec![pbr.sample_attrs.len(), 6],
                        sample_attrs,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.texture.base_color_float",
                        vec![pbr.texture_height, pbr.texture_width, 4],
                        base_color_float,
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.texture.metallic_float",
                        vec![pbr.texture_height, pbr.texture_width],
                        pbr.metallic_float.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.texture.roughness_float",
                        vec![pbr.texture_height, pbr.texture_width],
                        pbr.roughness_float.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        "pbr.texture.alpha_float",
                        vec![pbr.texture_height, pbr.texture_width],
                        pbr.alpha_float.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_u8(
                        "pbr.texture.base_color_rgba_u8",
                        vec![pbr.texture_height, pbr.texture_width, 4],
                        pbr.base_color_rgba_u8.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_u8(
                        "pbr.texture.metallic_roughness_u8",
                        vec![pbr.texture_height, pbr.texture_width, 4],
                        pbr.metallic_roughness_u8.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }
            trace.save(hook_output).map_err(TrellisRuntimeError::new)?;
        }
        Ok(())
    }
}

fn extract_sparse_row_noise_override(
    snapshot: &HookSnapshot,
    prefix: &str,
    source_path: &Path,
) -> Result<Option<SparseRowNoiseOverride>, TrellisRuntimeError> {
    let coords_key = format!("{prefix}.coords");
    let feats_key = format!("{prefix}.feats");
    let coords = snapshot.tensors.get(&coords_key);
    let feats = snapshot.tensors.get(&feats_key);
    match (coords, feats) {
        (None, None) => Ok(None),
        (Some(_), None) | (None, Some(_)) => Err(TrellisRuntimeError::new(format!(
            "noise override hook '{}' has incomplete sparse override for '{}': expected both '{}' and '{}'",
            source_path.display(),
            prefix,
            coords_key,
            feats_key
        ))),
        (Some(coords), Some(feats)) => {
            let mut coords_rows = hook_tensor_to_coords4(coords_key.as_str(), coords)?;
            let mut feat_rows = hook_tensor_to_rows32(feats_key.as_str(), feats)?;
            if coords_rows.len() != feat_rows.len() {
                let keep = coords_rows.len().min(feat_rows.len());
                coords_rows.truncate(keep);
                feat_rows.truncate(keep);
            }
            Ok(Some(SparseRowNoiseOverride {
                coords: coords_rows,
                feats: feat_rows,
            }))
        }
    }
}

fn extract_dense_f32_override(snapshot: &HookSnapshot, key: &str) -> Option<Vec<f32>> {
    snapshot.tensors.get(key).map(|tensor| tensor.data.clone())
}

fn extract_sampler_override(
    snapshot: &HookSnapshot,
    key: &str,
    source_path: &Path,
) -> Result<Option<SamplerConfigOverride>, TrellisRuntimeError> {
    let Some(tensor) = snapshot.tensors.get(key) else {
        return Ok(None);
    };
    if tensor.shape.len() != 1 || tensor.shape[0] != 7 {
        return Err(TrellisRuntimeError::new(format!(
            "sampler override hook '{}' key '{}' has invalid shape {:?}; expected [7]",
            source_path.display(),
            key,
            tensor.shape
        )));
    }
    if tensor.data.len() != 7 {
        return Err(TrellisRuntimeError::new(format!(
            "sampler override hook '{}' key '{}' has {} elements, expected 7",
            source_path.display(),
            key,
            tensor.data.len()
        )));
    }

    let steps = tensor.data[0].round().max(1.0) as usize;
    let rescale_t = tensor.data[1].max(f32::EPSILON);
    let guidance_strength = tensor.data[2];
    let guidance_rescale = tensor.data[3].max(0.0);
    let interval_start = tensor.data[4];
    let interval_end = tensor.data[5];
    let sigma_min = tensor.data[6].max(f32::EPSILON);

    Ok(Some(SamplerConfigOverride {
        sigma_min,
        config: FlowEulerSampleConfig {
            steps,
            rescale_t,
            guidance_strength,
            guidance_rescale,
            guidance_interval: [interval_start, interval_end],
        },
    }))
}

fn hook_tensor_to_coords4(
    key: &str,
    tensor: &HookTensor,
) -> Result<Vec<[u32; 4]>, TrellisRuntimeError> {
    if tensor.shape.len() != 2 || tensor.shape[1] != 4 {
        return Err(TrellisRuntimeError::new(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 4]",
            tensor.shape
        )));
    }
    let rows = tensor.shape[0];
    if tensor.data.len() != rows * 4 {
        return Err(TrellisRuntimeError::new(format!(
            "hook tensor '{key}' has {} elements, expected {}",
            tensor.data.len(),
            rows * 4
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 4;
        out.push([
            tensor.data[base].round().max(0.0) as u32,
            tensor.data[base + 1].round().max(0.0) as u32,
            tensor.data[base + 2].round().max(0.0) as u32,
            tensor.data[base + 3].round().max(0.0) as u32,
        ]);
    }
    Ok(out)
}

fn hook_tensor_to_rows32(
    key: &str,
    tensor: &HookTensor,
) -> Result<Vec<[f32; 32]>, TrellisRuntimeError> {
    if tensor.shape.len() != 2 || tensor.shape[1] != 32 {
        return Err(TrellisRuntimeError::new(format!(
            "hook tensor '{key}' has invalid shape {:?}; expected [N, 32]",
            tensor.shape
        )));
    }
    let rows = tensor.shape[0];
    if tensor.data.len() != rows * 32 {
        return Err(TrellisRuntimeError::new(format!(
            "hook tensor '{key}' has {} elements, expected {}",
            tensor.data.len(),
            rows * 32
        )));
    }
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 32;
        let mut values = [0.0f32; 32];
        values.copy_from_slice(&tensor.data[base..base + 32]);
        out.push(values);
    }
    Ok(out)
}

fn final_resolution_for_pipeline(pipeline_type: &str) -> usize {
    match pipeline_type {
        "512" | "512_base" => 512,
        "1024" | "1024_single" | "1024_cascade" => 1024,
        "1536_cascade" => 1536,
        _ => 512,
    }
}

fn build_synthetic_cond_trace(
    preprocess: &PreprocessOutput,
    tokens: usize,
    cond_channels: usize,
) -> Vec<f32> {
    if preprocess.rgb.is_empty() {
        return vec![0.0; tokens * cond_channels];
    }
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

fn insert_sampler_hook_config(
    trace: &mut HookTrace,
    prefix: &str,
    config: FlowEulerSampleConfig,
    sigma_min: f32,
) -> Result<(), String> {
    trace.insert_f32(
        format!("{prefix}.config"),
        vec![7],
        vec![
            config.steps as f32,
            config.rescale_t,
            config.guidance_strength,
            config.guidance_rescale,
            config.guidance_interval[0],
            config.guidance_interval[1],
            sigma_min,
        ],
    )?;
    let pairs = timestep_pairs(config.steps, config.rescale_t);
    let values = pairs
        .iter()
        .flat_map(|(t, t_prev)| [*t, *t_prev])
        .collect::<Vec<_>>();
    trace.insert_f32(
        format!("{prefix}.timestep_pairs"),
        vec![pairs.len(), 2],
        values,
    )?;
    Ok(())
}

fn insert_sparse_trace_rows(
    trace: &mut HookTrace,
    prefix: &str,
    coords: &[[u32; 4]],
    features: &[[f32; 32]],
    channels: usize,
    spatial_resolution: usize,
) -> Result<(), String> {
    let rows_total = coords.len().min(features.len());
    let row_indices = sampled_row_indices(rows_total, channels.max(1));
    let rows = row_indices.len();
    trace.insert_f32(
        format!("{prefix}.coords"),
        vec![rows, 4],
        flatten_coords_indices(coords, row_indices.as_slice()),
    )?;
    trace.insert_f32(
        format!("{prefix}.feats"),
        vec![rows, channels],
        flatten_feature_rows_indices(features, row_indices.as_slice(), channels),
    )?;
    trace.insert_f32(
        format!("{prefix}.shape"),
        vec![2],
        vec![1.0, channels as f32],
    )?;
    trace.insert_f32(
        format!("{prefix}.spatial_shape"),
        vec![3],
        vec![
            spatial_resolution as f32,
            spatial_resolution as f32,
            spatial_resolution as f32,
        ],
    )?;
    Ok(())
}

fn sparse_step_values(
    sparse: &crate::staged_pipeline::SparseStructureSample,
    step_idx: usize,
) -> Vec<f32> {
    let num_steps = sparse.step_count.max(1);
    let last_step = num_steps.saturating_sub(1);
    let mid_step = sampler_mid_step(num_steps);
    if step_idx == 0 {
        return sparse.step_0_x_t.clone();
    }
    if step_idx == mid_step {
        return sparse.step_mid_x_t.clone();
    }
    if step_idx == last_step {
        return sparse.step_last_x_t.clone();
    }
    sparse.step_last_x_t.clone()
}

fn shape_slat_step_values(
    shape: &crate::staged_pipeline::ShapeSLatSample,
    step_idx: usize,
) -> &[[f32; 32]] {
    let num_steps = shape.step_count.max(1);
    let last_step = num_steps.saturating_sub(1);
    let mid_step = sampler_mid_step(num_steps);
    if step_idx == 0 {
        return shape.step_0_x_t.as_slice();
    }
    if step_idx == mid_step {
        return shape.step_mid_x_t.as_slice();
    }
    if step_idx == last_step {
        return shape.step_last_x_t.as_slice();
    }
    shape.step_last_x_t.as_slice()
}

fn tex_slat_step_values(
    tex: &crate::staged_pipeline::TexSLatSample,
    step_idx: usize,
) -> &[[f32; 32]] {
    let num_steps = tex.step_count.max(1);
    let last_step = num_steps.saturating_sub(1);
    let mid_step = sampler_mid_step(num_steps);
    if step_idx == 0 {
        return tex.step_0_x_t.as_slice();
    }
    if step_idx == mid_step {
        return tex.step_mid_x_t.as_slice();
    }
    if step_idx == last_step {
        return tex.step_last_x_t.as_slice();
    }
    tex.step_last_x_t.as_slice()
}

fn sampler_snapshot_steps(num_steps: usize, snapshots: usize) -> Vec<usize> {
    let steps = num_steps.max(1);
    let snaps = snapshots.max(1);
    if steps <= snaps {
        return (0..steps).collect();
    }
    if snaps <= 1 {
        return vec![steps - 1];
    }

    let mut out = Vec::with_capacity(snaps);
    for i in 0..snaps {
        let pos = i as f64 * (steps - 1) as f64 / (snaps - 1) as f64;
        let idx = pos.round() as usize;
        if out.last().copied() != Some(idx) {
            out.push(idx);
        }
    }
    out
}

fn sampler_mid_step(num_steps: usize) -> usize {
    if num_steps <= 1 {
        return 0;
    }
    ((num_steps - 1) as f32 * 0.5).round() as usize
}

fn sample_dense_u8_for_hook(shape: &[usize], data: &[u8]) -> (Vec<usize>, Vec<u8>) {
    if hook_full_capture_enabled() {
        return (shape.to_vec(), data.to_vec());
    }

    let max_dense = hook_max_dense_elements();
    if shape.is_empty() {
        return (vec![data.len()], data.to_vec());
    }
    let mut sampled_shape = shape.to_vec();
    let mut sampled_data = data.to_vec();

    if sampled_shape.len() >= 2 && sampled_shape[0] > 0 {
        let per_row = sampled_shape[1..]
            .iter()
            .copied()
            .fold(1usize, |acc, dim| acc.saturating_mul(dim.max(1)));
        let row_cap = sampled_row_cap(per_row);
        let row_indices = uniform_sample_indices(sampled_shape[0], row_cap);
        if row_indices.len() != sampled_shape[0] {
            let mut row_sampled = Vec::with_capacity(row_indices.len().saturating_mul(per_row));
            for idx in row_indices {
                let start = idx.saturating_mul(per_row);
                let end = start.saturating_add(per_row).min(sampled_data.len());
                if end > start {
                    row_sampled.extend_from_slice(&sampled_data[start..end]);
                }
            }
            sampled_shape[0] = row_sampled.len() / per_row.max(1);
            sampled_data = row_sampled;
        }
    }

    let numel = sampled_shape
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim.max(1)));
    if numel > max_dense && !sampled_data.is_empty() {
        let flat_indices = uniform_sample_indices(numel, max_dense);
        let mut flat = Vec::with_capacity(flat_indices.len());
        for idx in flat_indices {
            if let Some(value) = sampled_data.get(idx) {
                flat.push(*value);
            }
        }
        sampled_shape = vec![flat.len()];
        sampled_data = flat;
    }

    (sampled_shape, sampled_data)
}

fn sample_dense_f32_for_hook(shape: &[usize], data: &[f32]) -> (Vec<usize>, Vec<f32>) {
    if hook_full_capture_enabled() {
        return (shape.to_vec(), data.to_vec());
    }

    let max_dense = hook_max_dense_elements();
    if shape.is_empty() {
        return (vec![data.len()], data.to_vec());
    }
    let mut sampled_shape = shape.to_vec();
    let mut sampled_data = data.to_vec();

    if sampled_shape.len() >= 2 && sampled_shape[0] > 0 {
        let per_row = sampled_shape[1..]
            .iter()
            .copied()
            .fold(1usize, |acc, dim| acc.saturating_mul(dim.max(1)));
        let row_cap = sampled_row_cap(per_row);
        let row_indices = uniform_sample_indices(sampled_shape[0], row_cap);
        if row_indices.len() != sampled_shape[0] {
            let mut row_sampled = Vec::with_capacity(row_indices.len().saturating_mul(per_row));
            for idx in row_indices {
                let start = idx.saturating_mul(per_row);
                let end = start.saturating_add(per_row).min(sampled_data.len());
                if end > start {
                    row_sampled.extend_from_slice(&sampled_data[start..end]);
                }
            }
            sampled_shape[0] = row_sampled.len() / per_row.max(1);
            sampled_data = row_sampled;
        }
    }

    let numel = sampled_shape
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim.max(1)));
    if numel > max_dense && !sampled_data.is_empty() {
        let flat_indices = uniform_sample_indices(numel, max_dense);
        let mut flat = Vec::with_capacity(flat_indices.len());
        for idx in flat_indices {
            if let Some(value) = sampled_data.get(idx) {
                flat.push(*value);
            }
        }
        sampled_shape = vec![flat.len()];
        sampled_data = flat;
    }

    (sampled_shape, sampled_data)
}

fn sampled_row_cap(per_row_elements: usize) -> usize {
    if hook_full_capture_enabled() {
        return usize::MAX;
    }
    let dense_cap = (hook_max_dense_elements() / per_row_elements.max(1)).max(1);
    hook_max_rows().min(dense_cap).max(1)
}

fn sampled_row_indices(total_rows: usize, per_row_elements: usize) -> Vec<usize> {
    if total_rows == 0 {
        return Vec::new();
    }
    let cap = sampled_row_cap(per_row_elements);
    uniform_sample_indices(total_rows, cap)
}

fn uniform_sample_indices(total: usize, cap: usize) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    if total <= cap {
        return (0..total).collect();
    }
    if cap <= 1 {
        return vec![total - 1];
    }
    let mut out = Vec::with_capacity(cap);
    let mut last = usize::MAX;
    for i in 0..cap {
        let pos = i as f64 * (total - 1) as f64 / (cap - 1) as f64;
        let idx = pos.round() as usize;
        if idx != last {
            out.push(idx);
            last = idx;
        }
    }
    out
}

fn flatten_coords_indices(coords: &[[u32; 4]], indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * 4);
    for idx in indices {
        let coord = coords[*idx];
        out.push(coord[0] as f32);
        out.push(coord[1] as f32);
        out.push(coord[2] as f32);
        out.push(coord[3] as f32);
    }
    out
}

fn flatten_coords3_indices(coords: &[[u32; 4]], indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * 3);
    for idx in indices {
        let coord = coords[*idx];
        out.push(coord[1] as f32);
        out.push(coord[2] as f32);
        out.push(coord[3] as f32);
    }
    out
}

fn flatten_feature_rows_indices(
    rows: &[[f32; 32]],
    indices: &[usize],
    channels: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * channels);
    for idx in indices {
        let row = rows[*idx];
        out.extend(row.iter().take(channels).copied());
    }
    out
}

fn flatten_feature_rows_indices_fixed<const C: usize>(
    rows: &[[f32; C]],
    indices: &[usize],
    channels: usize,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * channels);
    for idx in indices {
        let row = rows[*idx];
        out.extend(row.iter().take(channels).copied());
    }
    out
}

fn flatten_vertices_indices(vertices: &[[f32; 3]], indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * 3);
    for idx in indices {
        let vertex = vertices[*idx];
        out.extend_from_slice(&vertex);
    }
    out
}

fn flatten_faces_indices(faces: &[[u32; 3]], indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len() * 3);
    for idx in indices {
        let face = faces[*idx];
        out.push(face[0] as f32);
        out.push(face[1] as f32);
        out.push(face[2] as f32);
    }
    out
}

fn collect_model_stems(pipeline_json: &Value) -> Vec<String> {
    let mut stems = Vec::new();
    if let Some(models) = pipeline_json
        .get("args")
        .and_then(|value| value.get("models"))
        .and_then(Value::as_object)
    {
        for value in models.values() {
            if let Some(stem) = value.as_str() {
                stems.push(stem.to_string());
            }
        }
    }
    stems.sort();
    stems.dedup();
    stems
}

fn resolve_model_source_path(
    stem: &str,
    ext: &str,
    weights_root: &Path,
    image_large_root: Option<&Path>,
) -> PathBuf {
    if stem.starts_with("ckpts/") {
        return weights_root.join(format!("{stem}.{ext}"));
    }
    if let Some((_, suffix)) = stem.split_once("/ckpts/") {
        let image_large_root = image_large_root.unwrap_or(weights_root);
        return image_large_root.join(format!("ckpts/{suffix}.{ext}"));
    }
    weights_root.join(format!("{stem}.{ext}"))
}

fn with_file_stem_suffix(path: &Path, suffix: &str) -> PathBuf {
    let Some(stem) = path.file_stem() else {
        return path.to_path_buf();
    };
    let stem = stem.to_string_lossy();
    if stem.ends_with(suffix) {
        return path.to_path_buf();
    }
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    let mut file_name = format!("{stem}{suffix}");
    if !ext.is_empty() {
        file_name.push('.');
        file_name.push_str(ext);
    }
    path.with_file_name(file_name)
}

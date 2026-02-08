use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::ValueEnum;
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::TrellisQuality;
use crate::mesh::{Mesh, write_obj_mesh};
use crate::native::hook_trace::HookTrace;
use crate::native::preprocess::{PreprocessConfig, preprocess_image, preprocess_image_path};
use crate::paths::{resolve_trellis2_image_large_root, resolve_trellis2_weights_root};
use crate::staged_pipeline::{SparseStructureStageSource, TrellisStageOutput, TrellisStageRuntime};
use crate::trellis_config::TrellisPipelineConfig;

const HOOK_MAX_ROWS: usize = 1024;
const HOOK_MAX_DENSE_ELEMENTS: usize = 200_000;

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

#[derive(Clone, Debug)]
pub struct TrellisRunOptions {
    pub quality: TrellisQuality,
    pub device: TrellisDevice,
    pub seed: Option<u64>,
    pub hook_output: Option<PathBuf>,
}

impl Default for TrellisRunOptions {
    fn default() -> Self {
        Self {
            quality: TrellisQuality::default(),
            device: TrellisDevice::default(),
            seed: None,
            hook_output: None,
        }
    }
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
        Self {
            weights_root: resolve_trellis2_weights_root(None),
            image_large_root: Some(resolve_trellis2_image_large_root(None)),
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
        let mut config = Trellis2PipelineConfig::default();
        config.weights_root = weights_root;
        Self::new(config)
    }

    pub fn new(config: Trellis2PipelineConfig) -> Result<Self, TrellisRuntimeError> {
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
        let (stage_output, stage_timings) = runtime.run_profiled(&preprocess, seed);
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
        let stage_output = runtime.run(&preprocess, seed);
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

    fn capture_pipeline_hook(
        &self,
        preprocess: &crate::native::preprocess::PreprocessOutput,
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
            trace
                .insert_u8(
                    "preprocess_image.output",
                    preprocess_shape.clone(),
                    preprocess.rgb.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_u8("run.image", preprocess_shape, preprocess.rgb.clone())
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

            let cond = build_synthetic_cond_trace(preprocess, HOOK_MAX_DENSE_ELEMENTS);
            trace
                .insert_f32("get_cond_512.out.cond", vec![cond.len()], cond)
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "get_cond_512.out.neg_cond",
                    vec![HOOK_MAX_DENSE_ELEMENTS],
                    vec![0.0f32; HOOK_MAX_DENSE_ELEMENTS],
                )
                .map_err(TrellisRuntimeError::new)?;

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
            trace
                .insert_f32(
                    "sample_sparse_structure.sampler.step_000_of_012.x_t",
                    sparse_shape.clone(),
                    stage_output.sparse.step_0_x_t.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
            trace
                .insert_f32(
                    "sample_sparse_structure.sampler.step_011_of_012.x_t",
                    sparse_shape.clone(),
                    stage_output.sparse.step_last_x_t.clone(),
                )
                .map_err(TrellisRuntimeError::new)?;
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

            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.noise",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.noise,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.sampler.step_000_of_012.x_t",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.step_0_x_t,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.sampler.step_011_of_012.x_t",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.step_last_x_t,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_shape_slat.slat",
                &stage_output.shape_slat.coords,
                &stage_output.shape_slat.features,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;

            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.noise",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.noise,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.sampler.step_000_of_012.x_t",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.step_0_x_t,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
            insert_sparse_trace_rows(
                &mut trace,
                "sample_tex_slat.sampler.step_011_of_012.x_t",
                &stage_output.tex_slat.coords,
                &stage_output.tex_slat.step_last_x_t,
                32,
                stage_output.sparse.resolution,
            )
            .map_err(TrellisRuntimeError::new)?;
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

            let fallback_subs_indices =
                sampled_row_indices(stage_output.shape_slat.coords.len(), 4);
            let fallback_subs_rows = fallback_subs_indices.len();
            let fallback_subs_coords = flatten_coords_indices(
                &stage_output.shape_slat.coords,
                fallback_subs_indices.as_slice(),
            );
            let mut fallback_subs_feats = Vec::with_capacity(fallback_subs_rows * 8);
            for idx in fallback_subs_indices {
                if let Some(row) = stage_output.shape_slat.features.get(idx) {
                    fallback_subs_feats.extend(row.iter().take(8).copied());
                }
            }
            for level in 0..4usize {
                let prefix = format!("decode_shape_slat.subs.{level}");
                if let Some(sub) = stage_output.decode_shape_subs.get(level) {
                    let mut sub_indices = sampled_row_indices(sub.coords.len(), 4);
                    pad_indices_to(&mut sub_indices, HOOK_MAX_ROWS);
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
                    .insert_f32(
                        format!("{prefix}.coords"),
                        vec![fallback_subs_rows, 4],
                        fallback_subs_coords.clone(),
                    )
                    .map_err(TrellisRuntimeError::new)?;
                trace
                    .insert_f32(
                        format!("{prefix}.feats"),
                        vec![fallback_subs_rows, 8],
                        fallback_subs_feats.clone(),
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
                            (32usize << level) as f32,
                            (32usize << level) as f32,
                            (32usize << level) as f32,
                        ],
                    )
                    .map_err(TrellisRuntimeError::new)?;
            }

            let voxel_source_coords = if stage_output.decode_tex_voxels.coords.is_empty() {
                &stage_output.tex_slat.coords
            } else {
                &stage_output.decode_tex_voxels.coords
            };
            let voxel_indices = sampled_row_indices(voxel_source_coords.len(), 4);
            let voxel_rows = voxel_indices.len();
            let mut voxel_attrs = Vec::with_capacity(voxel_rows * 6);
            for idx in voxel_indices.iter().copied() {
                if let Some(row) = stage_output.decode_tex_voxels.feats.get(idx) {
                    voxel_attrs.extend(row.iter().copied());
                } else if let Some(row) = stage_output.tex_slat.features.get(idx) {
                    voxel_attrs.extend(row.iter().take(6).copied());
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
            trace.save(hook_output).map_err(TrellisRuntimeError::new)?;
        }
        Ok(())
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

fn build_synthetic_cond_trace(
    preprocess: &crate::native::preprocess::PreprocessOutput,
    max_elements: usize,
) -> Vec<f32> {
    if preprocess.rgb.is_empty() {
        return vec![0.0; max_elements];
    }
    let mut out = Vec::with_capacity(max_elements);
    for idx in 0..max_elements {
        let value = preprocess.rgb[idx % preprocess.rgb.len()] as f32 / 255.0;
        let scale = 0.25 + ((idx % 257) as f32 / 257.0);
        out.push((value * scale).clamp(0.0, 1.0));
    }
    out
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

fn sampled_row_indices(total_rows: usize, per_row_elements: usize) -> Vec<usize> {
    if total_rows == 0 {
        return Vec::new();
    }
    let dense_cap = (HOOK_MAX_DENSE_ELEMENTS / per_row_elements.max(1)).max(1);
    let cap = HOOK_MAX_ROWS.min(dense_cap).max(1);
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

fn pad_indices_to(indices: &mut Vec<usize>, target: usize) {
    if indices.is_empty() || indices.len() >= target {
        return;
    }
    let seed = indices.clone();
    while indices.len() < target {
        indices.push(seed[indices.len() % seed.len()]);
    }
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

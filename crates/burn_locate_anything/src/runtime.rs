use std::path::PathBuf;

use image::DynamicImage;
use serde::{Deserialize, Serialize};

use crate::assets::{LocateAnythingAssetReport, inspect_model_assets};
use crate::config::LocateAnythingModelConfig;
use crate::decode::{DecodeMode, decode_detections_from_text};
use crate::native::{LocateAnythingNativeBatchInputs, prepare_native_batch_inputs};
use crate::tokenizer::grounding_prompt;
use crate::vision::LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT;
use crate::{Detection, DetectionQuery, LocateAnythingError, LocateAnythingResult};

#[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
struct WgpuNativeRuntimeCache {
    device: burn_wgpu::WgpuDevice,
    vision: crate::native::burn_native::BurnLocateAnythingVisionProjector<
        burn_wgpu::Wgpu<f32, i32, u32>,
    >,
    qwen: crate::native::burn_native::BurnLocateAnythingQwen<burn_wgpu::Wgpu<f32, i32, u32>>,
    tokenizer: crate::tokenizer::QwenTokenizer,
}

#[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
impl std::fmt::Debug for WgpuNativeRuntimeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuNativeRuntimeCache")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LocateAnythingRuntimeBackend {
    /// Native Burn graph. Detection remains explicitly opt-in until the full
    /// MoonViT + Qwen + PBD stack has broad hook-validated checkpoint parity.
    #[default]
    BurnNative,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct LocateAnythingRuntimeConfig {
    pub model_root: PathBuf,
    pub backend: LocateAnythingRuntimeBackend,
    #[serde(default)]
    pub allow_experimental_native_detect: bool,
    pub decode_mode: DecodeMode,
    pub max_new_tokens: usize,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
    #[serde(default = "default_top_p")]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<usize>,
    pub batch_prompts: bool,
    pub require_gpu: bool,
    pub in_token_limit: u32,
}

fn default_repetition_penalty() -> f32 {
    1.1
}

fn default_top_p() -> Option<f32> {
    Some(0.9)
}

impl Default for LocateAnythingRuntimeConfig {
    fn default() -> Self {
        Self {
            model_root: PathBuf::from("assets/models/LocateAnything-3B"),
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: false,
            decode_mode: DecodeMode::Hybrid,
            max_new_tokens: 8192,
            repetition_penalty: default_repetition_penalty(),
            top_p: default_top_p(),
            top_k: None,
            batch_prompts: true,
            require_gpu: true,
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BatchedDetectionRequest {
    pub image_path: PathBuf,
    pub queries: Vec<DetectionQuery>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LocateAnythingBurnNativeStageTimings {
    pub total_ms: f64,
    pub prepare_ms: f64,
    pub cache_init_ms: f64,
    pub runtime_cache_hit: bool,
    pub vision_projector_ms: f64,
    pub queries: Vec<LocateAnythingBurnNativeQueryTiming>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct LocateAnythingBurnNativeQueryTiming {
    pub query: String,
    pub generate_ms: f64,
    pub decode_ms: f64,
    pub generated_tokens: usize,
    pub detections: usize,
    pub used_batched_initial_step: bool,
}

pub trait LocateAnythingDetector {
    fn detect(
        &mut self,
        image: &DynamicImage,
        query: &DetectionQuery,
    ) -> LocateAnythingResult<Vec<Detection>>;

    fn detect_batch(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        queries
            .iter()
            .map(|query| self.detect(image, query))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct LocateAnythingRuntime {
    config: LocateAnythingRuntimeConfig,
    model_config: Option<LocateAnythingModelConfig>,
    asset_report: LocateAnythingAssetReport,
    last_burn_native_stage_timings: Option<LocateAnythingBurnNativeStageTimings>,
    #[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
    native_wgpu_cache: std::sync::Arc<std::sync::Mutex<Option<WgpuNativeRuntimeCache>>>,
}

impl LocateAnythingRuntime {
    pub fn new(config: LocateAnythingRuntimeConfig) -> LocateAnythingResult<Self> {
        if config.max_new_tokens == 0 {
            return Err(LocateAnythingError::Config(
                "max_new_tokens must be greater than zero".to_string(),
            ));
        }
        let asset_report = inspect_model_assets(&config.model_root)?;
        let model_config = if asset_report.config_present {
            Some(LocateAnythingModelConfig::from_model_root(
                &config.model_root,
            )?)
        } else {
            None
        };
        Ok(Self {
            config,
            model_config,
            asset_report,
            last_burn_native_stage_timings: None,
            #[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
            native_wgpu_cache: Default::default(),
        })
    }

    pub fn config(&self) -> &LocateAnythingRuntimeConfig {
        &self.config
    }

    pub fn model_config(&self) -> Option<&LocateAnythingModelConfig> {
        self.model_config.as_ref()
    }

    pub fn asset_report(&self) -> &LocateAnythingAssetReport {
        &self.asset_report
    }

    pub fn last_burn_native_stage_timings(&self) -> Option<&LocateAnythingBurnNativeStageTimings> {
        self.last_burn_native_stage_timings.as_ref()
    }

    pub fn decode_fixture_output(
        &self,
        query: impl Into<String>,
        text: &str,
    ) -> LocateAnythingResult<Vec<Detection>> {
        decode_detections_from_text(query, text)
    }

    pub fn prompt_for_query(&self, query: &DetectionQuery) -> String {
        grounding_prompt(&query.query)
    }

    pub fn prepare_native_batch_inputs(
        &self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<LocateAnythingNativeBatchInputs> {
        let model_config = self.model_config.as_ref().ok_or_else(|| {
            LocateAnythingError::Config(format!(
                "LocateAnything config.json is required for native preparation under {}",
                self.config.model_root.display()
            ))
        })?;
        prepare_native_batch_inputs(
            &self.config.model_root,
            model_config,
            self.config.in_token_limit,
            image,
            queries,
        )
    }
}

impl LocateAnythingDetector for LocateAnythingRuntime {
    fn detect(
        &mut self,
        image: &DynamicImage,
        query: &DetectionQuery,
    ) -> LocateAnythingResult<Vec<Detection>> {
        let mut batch = self.detect_batch(image, std::slice::from_ref(query))?;
        Ok(batch.remove(0))
    }

    fn detect_batch(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        if !self.asset_report.is_complete() {
            return Err(LocateAnythingError::Runtime(format!(
                "LocateAnything model assets are incomplete under {}; missing files: {:?}",
                self.config.model_root.display(),
                self.asset_report.missing_files
            )));
        }
        if queries.is_empty() {
            return Ok(Vec::new());
        }
        let LocateAnythingRuntimeBackend::BurnNative = self.config.backend;
        if !self.config.allow_experimental_native_detect {
            return self.unsupported_native_boundary(image, queries);
        }
        self.detect_batch_burn_native(image, queries)
    }
}

impl LocateAnythingRuntime {
    #[cfg(all(feature = "backend_wgpu", not(target_arch = "wasm32")))]
    fn detect_batch_burn_native(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        use crate::native::burn_native::{
            BurnLocateAnythingQwen, BurnLocateAnythingVisionProjector, BurnQwenGenerationConfig,
        };

        let model_config = self.model_config.as_ref().ok_or_else(|| {
            LocateAnythingError::Config(format!(
                "LocateAnything config.json is required for native detection under {}",
                self.config.model_root.display()
            ))
        })?;
        let total_started = std::time::Instant::now();
        let prepare_started = std::time::Instant::now();
        let prepared = self.prepare_native_batch_inputs(image, queries)?;
        let prepare_ms = prepare_started.elapsed().as_secs_f64() * 1000.0;

        let (results, mut timings) = {
            let mut cache = self.native_wgpu_cache.lock().map_err(|err| {
                LocateAnythingError::Runtime(format!(
                    "failed to lock LocateAnything WGPU runtime cache: {err}"
                ))
            })?;
            let runtime_cache_hit = cache.is_some();
            let cache_init_started = std::time::Instant::now();
            if cache.is_none() {
                let device = burn_wgpu::WgpuDevice::default();
                let vision = BurnLocateAnythingVisionProjector::<
                    burn_wgpu::Wgpu<f32, i32, u32>,
                >::from_model_root(
                    &self.config.model_root, model_config, &device
                )?;
                let qwen =
                    BurnLocateAnythingQwen::<burn_wgpu::Wgpu<f32, i32, u32>>::from_model_root(
                        &self.config.model_root,
                        model_config,
                        &device,
                    )?;
                let tokenizer =
                    crate::tokenizer::QwenTokenizer::from_model_root(&self.config.model_root)?;
                *cache = Some(WgpuNativeRuntimeCache {
                    device,
                    vision,
                    qwen,
                    tokenizer,
                });
            }
            let cache_init_ms = cache_init_started.elapsed().as_secs_f64() * 1000.0;
            let cache = cache.as_ref().expect("cache initialized above");
            let vision_started = std::time::Instant::now();
            let image_features = cache
                .vision
                .forward_preprocessed(&prepared.image, &cache.device);
            let vision_projector_ms = vision_started.elapsed().as_secs_f64() * 1000.0;

            let mut query_timings = Vec::with_capacity(prepared.prompts.len());
            let mut results = Vec::with_capacity(prepared.prompts.len());
            let prompt_refs = prepared
                .prompts
                .iter()
                .map(|prompt| &prompt.prompt)
                .collect::<Vec<_>>();
            let generated_batch = cache.qwen.generate_token_ids_batch_with_config(
                &prompt_refs,
                image_features.clone(),
                BurnQwenGenerationConfig {
                    decode_mode: self.config.decode_mode,
                    max_new_tokens: self.config.max_new_tokens,
                    repetition_penalty: self.config.repetition_penalty,
                    top_p: self.config.top_p,
                    top_k: self.config.top_k,
                },
                &cache.device,
            )?;
            for (prompt, generated) in prepared.prompts.iter().zip(generated_batch) {
                let decode_started = std::time::Instant::now();
                let answer = cache.tokenizer.decode(&generated.token_ids, false)?;
                let detections = decode_detections_from_text(prompt.query.query.clone(), &answer)?;
                let decode_ms = decode_started.elapsed().as_secs_f64() * 1000.0;
                query_timings.push(LocateAnythingBurnNativeQueryTiming {
                    query: prompt.query.query.clone(),
                    generate_ms: generated.elapsed_ms,
                    decode_ms,
                    generated_tokens: generated.token_ids.len(),
                    detections: detections.len(),
                    used_batched_initial_step: generated.used_batched_initial_step,
                });
                results.push(detections);
            }
            (
                results,
                LocateAnythingBurnNativeStageTimings {
                    total_ms: 0.0,
                    prepare_ms,
                    cache_init_ms,
                    runtime_cache_hit,
                    vision_projector_ms,
                    queries: query_timings,
                },
            )
        };
        timings.total_ms = total_started.elapsed().as_secs_f64() * 1000.0;
        self.last_burn_native_stage_timings = Some(timings);
        Ok(results)
    }

    #[cfg(all(
        not(feature = "backend_wgpu"),
        feature = "backend_cuda",
        not(target_arch = "wasm32")
    ))]
    fn detect_batch_burn_native(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        let device = burn_cuda::CudaDevice::default();
        self.detect_batch_burn_native_backend::<burn_cuda::Cuda<f32, i32>>(&device, image, queries)
    }

    #[cfg(all(
        not(feature = "backend_wgpu"),
        not(feature = "backend_cuda"),
        feature = "backend_ndarray",
        not(target_arch = "wasm32")
    ))]
    fn detect_batch_burn_native(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        use burn::tensor::backend::BackendTypes;

        let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
        self.detect_batch_burn_native_backend::<burn::backend::NdArray<f32>>(
            &device, image, queries,
        )
    }

    #[cfg(any(
        target_arch = "wasm32",
        all(
            not(feature = "backend_wgpu"),
            not(feature = "backend_cuda"),
            not(feature = "backend_ndarray")
        )
    ))]
    fn detect_batch_burn_native(
        &mut self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        let _ = (image, queries);
        Err(LocateAnythingError::Unsupported(
            "Burn-native LocateAnything detect requires a native backend feature such as backend_wgpu, backend_cuda, or backend_ndarray; wasm native tokenizer/model execution is not enabled yet"
                .to_string(),
        ))
    }

    #[cfg(all(
        not(feature = "backend_wgpu"),
        any(feature = "backend_cuda", feature = "backend_ndarray"),
        not(target_arch = "wasm32")
    ))]
    fn detect_batch_burn_native_backend<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        use crate::native::burn_native::{
            BurnLocateAnythingQwen, BurnLocateAnythingVisionProjector, BurnQwenGenerationConfig,
        };
        use crate::tokenizer::QwenTokenizer;

        let model_config = self.model_config.as_ref().ok_or_else(|| {
            LocateAnythingError::Config(format!(
                "LocateAnything config.json is required for native detection under {}",
                self.config.model_root.display()
            ))
        })?;
        let prepared = self.prepare_native_batch_inputs(image, queries)?;
        let vision = BurnLocateAnythingVisionProjector::<B>::from_model_root(
            &self.config.model_root,
            model_config,
            device,
        )?;
        let image_features = vision.forward_preprocessed(&prepared.image, device);
        let qwen = BurnLocateAnythingQwen::<B>::from_model_root(
            &self.config.model_root,
            model_config,
            device,
        )?;
        let tokenizer = QwenTokenizer::from_model_root(&self.config.model_root)?;

        prepared
            .prompts
            .iter()
            .map(|prompt| {
                let generated = qwen.generate_token_ids_with_config(
                    &prompt.prompt,
                    image_features.clone(),
                    BurnQwenGenerationConfig {
                        decode_mode: self.config.decode_mode,
                        max_new_tokens: self.config.max_new_tokens,
                        repetition_penalty: self.config.repetition_penalty,
                        top_p: self.config.top_p,
                        top_k: self.config.top_k,
                    },
                    device,
                )?;
                let answer = tokenizer.decode(&generated, false)?;
                decode_detections_from_text(prompt.query.query.clone(), &answer)
            })
            .collect()
    }

    fn unsupported_native_boundary(
        &self,
        image: &DynamicImage,
        queries: &[DetectionQuery],
    ) -> LocateAnythingResult<Vec<Vec<Detection>>> {
        let prepared = self.prepare_native_batch_inputs(image, queries)?;
        let patch_tokens = prepared.image.patch_shape[0];
        let merged_tokens = prepared
            .prompts
            .first()
            .map(|prompt| prompt.prompt.image_context_tokens)
            .unwrap_or_default();
        let prompt_tokens = prepared
            .prompts
            .iter()
            .map(|prompt| prompt.prompt.input_ids.len())
            .collect::<Vec<_>>();
        let labels = queries
            .iter()
            .map(|query| query.query.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        Err(LocateAnythingError::Unsupported(format!(
            "Burn-native LocateAnything detect() for [{labels}] is opt-in while coverage broadens beyond the current WGPU hook fixtures: native preprocessing/tokenization/prompt assembly, MoonViT/projector, multimodal Qwen generation, top-p/repetition sampling, and PBD/hybrid decode are wired and validated on the Galaxy/table fixture; prepared patch_tokens={patch_tokens}, merged_image_tokens={merged_tokens}, prompt_tokens={prompt_tokens:?}. Set allow_experimental_native_detect=true for validated WGPU parity/debug runs."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn runtime_builds_prompt_for_query() {
        let runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig::default()).unwrap();
        let prompt = runtime.prompt_for_query(&DetectionQuery {
            query: "all chairs".to_string(),
            label_hint: None,
        });
        assert!(prompt.contains("all chairs"));
    }

    #[test]
    fn runtime_reports_asset_status() {
        let runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig::default()).unwrap();
        if !runtime.config().model_root.exists() {
            eprintln!(
                "skipping asset status assertion; {} is missing",
                runtime.config().model_root.display()
            );
            return;
        }
        assert!(runtime.asset_report().config_present);
    }

    #[test]
    fn runtime_fixture_decode_is_available_without_model_weights() {
        let runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig::default()).unwrap();
        let detections = runtime
            .decode_fixture_output("find tables", "table: <box>0.1, 0.2, 0.8, 0.7</box>")
            .unwrap();
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].bbox, [0.1, 0.2, 0.8, 0.7]);
    }

    #[test]
    fn runtime_burn_native_prepares_inputs_then_reports_graph_boundary_when_assets_present() {
        let Some(repo_root) = find_repo_root_for_test() else {
            eprintln!("skipping BurnNative boundary fixture; repo root not found");
            return;
        };
        let model_root = repo_root.join("assets/models/LocateAnything-3B");
        if !model_root.join("config.json").exists() {
            eprintln!(
                "skipping BurnNative boundary fixture; missing {}",
                model_root.display()
            );
            return;
        }
        let mut runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig {
            model_root,
            backend: LocateAnythingRuntimeBackend::BurnNative,
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            ..LocateAnythingRuntimeConfig::default()
        })
        .unwrap();
        let image = DynamicImage::new_rgb8(224, 224);
        let err = runtime
            .detect_batch(
                &image,
                &[DetectionQuery {
                    query: "plain square".to_string(),
                    label_hint: None,
                }],
            )
            .unwrap_err();
        let message = err.to_string();
        assert!(message.contains("is opt-in while coverage broadens"));
        assert!(message.contains("MoonViT/projector"));
        assert!(message.contains("top-p/repetition sampling"));
        assert!(message.contains("validated on the Galaxy/table fixture"));
        assert!(message.contains("patch_tokens="));
        assert!(message.contains("prompt_tokens="));
    }

    fn find_repo_root_for_test() -> Option<PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_locate_anything").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}

#[cfg(all(test, feature = "backend_wgpu", not(target_arch = "wasm32")))]
mod wgpu_runtime_tests {
    use super::*;

    #[test]
    fn burn_native_full_detect_matches_galaxy_reference_when_enabled() {
        if std::env::var("LOCATE_ANYTHING_BURN_NATIVE_FULL_DETECT_PARITY").is_err() {
            eprintln!(
                "skipping: set LOCATE_ANYTHING_BURN_NATIVE_FULL_DETECT_PARITY=1 to run full WGPU Burn-native LocateAnything detect parity"
            );
            return;
        }
        let Some(root) = find_repo_root_for_test() else {
            eprintln!("skipping full native detect parity; repo root not found");
            return;
        };
        let model_root = root.join("assets/models/LocateAnything-3B");
        let Some(image_path) =
            std::env::var_os("LOCATE_ANYTHING_PARITY_IMAGE").map(std::path::PathBuf::from)
        else {
            eprintln!(
                "skipping full native detect parity; set LOCATE_ANYTHING_PARITY_IMAGE to the reference scene image"
            );
            return;
        };
        if !model_root.join("config.json").exists() || !image_path.exists() {
            eprintln!(
                "skipping full native detect parity; missing {} or {}",
                model_root.display(),
                image_path.display()
            );
            return;
        }

        let mut runtime = LocateAnythingRuntime::new(LocateAnythingRuntimeConfig {
            model_root,
            backend: LocateAnythingRuntimeBackend::BurnNative,
            allow_experimental_native_detect: true,
            in_token_limit: LOCATE_ANYTHING_SAFE_IN_TOKEN_LIMIT,
            max_new_tokens: 128,
            ..LocateAnythingRuntimeConfig::default()
        })
        .unwrap();
        let image = image::open(&image_path).unwrap();
        let query = [
            DetectionQuery {
                query: "conference table".to_string(),
                label_hint: None,
            },
            DetectionQuery {
                query: "conference chair".to_string(),
                label_hint: None,
            },
        ];
        let cold_start = std::time::Instant::now();
        let detections = runtime.detect_batch(&image, &query).unwrap();
        let cold_ms = cold_start.elapsed().as_secs_f64() * 1000.0;
        let warm_start = std::time::Instant::now();
        let warm_detections = runtime.detect_batch(&image, &query).unwrap();
        let warm_ms = warm_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "LocateAnything Burn-native WGPU full detect parity timings: cold_ms={cold_ms:.2} warm_ms={warm_ms:.2}"
        );
        assert_eq!(detections, warm_detections);
        let bbox_tolerance = 3.0e-3;
        let expected = [
            vec![[0.386, 0.519, 0.659, 1.0]],
            vec![
                [0.166, 0.63, 0.36, 1.0],
                [0.311, 0.503, 0.412, 0.861],
                [0.381, 0.464, 0.442, 0.639],
                [0.478, 0.465, 0.532, 0.521],
                [0.574, 0.467, 0.626, 0.589],
                [0.612, 0.515, 0.673, 0.744],
                [0.613, 0.662, 0.839, 1.0],
                [0.781, 0.369, 0.833, 0.506],
            ],
        ];
        assert_eq!(detections.len(), expected.len());
        for (query_index, expected_boxes) in expected.iter().enumerate() {
            assert_eq!(detections[query_index].len(), expected_boxes.len());
            for (detection, expected_box) in detections[query_index].iter().zip(expected_boxes) {
                assert_eq!(detection.label, query[query_index].query);
                for (actual, expected) in detection.bbox.into_iter().zip(*expected_box) {
                    assert!(
                        (actual - expected).abs() <= bbox_tolerance,
                        "bbox mismatch for query {}: actual={:?}, expected={expected_box:?}",
                        query[query_index].query,
                        detection.bbox
                    );
                }
            }
        }
    }

    fn find_repo_root_for_test() -> Option<std::path::PathBuf> {
        let mut dir = std::env::current_dir().ok()?;
        loop {
            if dir.join("Cargo.toml").exists() && dir.join("crates/burn_locate_anything").exists() {
                return Some(dir);
            }
            if !dir.pop() {
                return None;
            }
        }
    }
}

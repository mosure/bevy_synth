#[cfg(not(target_arch = "wasm32"))]
use std::cmp::Reverse;
use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::SystemTime;

#[cfg(test)]
use burn::module::{Module, Param};
use burn::nn;
use burn::prelude::Backend;
#[cfg(test)]
use burn::tensor::Int;
use burn::tensor::Tensor;
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors, serialize};
use serde::Deserialize;

use super::weight_parts::{candidate_exists_or_has_parts, load_blob_bytes_from_burnpack_or_parts};
use crate::blob_burnpack::load_blob_bytes_from_burnpack as load_blob_bytes_from_blob_burnpack;
use crate::preprocess::PreprocessOutput;
use crate::virtual_fs;

type CpuRuntimeBackend = burn::backend::NdArray<f32>;
#[cfg(feature = "runtime-model-wgpu")]
type WgpuRuntimeBackend = burn_wgpu::Wgpu<f32, i32, u32>;

const DINO_IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const DINO_IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone)]
pub struct TrellisImageConditioningOutput {
    pub resolution: usize,
    pub token_count: usize,
    pub channels: usize,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub(crate) enum TrellisImageConditioningRuntime {
    Cpu(Box<TrellisImageConditioningRuntimeImpl<CpuRuntimeBackend>>),
    #[cfg(feature = "runtime-model-wgpu")]
    Wgpu(Box<TrellisImageConditioningRuntimeImpl<WgpuRuntimeBackend>>),
}

pub fn extract_condition_from_model_name(
    weights_root: &Path,
    image_large_root: Option<&Path>,
    model_name: &str,
    prefer_wgpu: bool,
    preprocess: &PreprocessOutput,
    resolution: usize,
) -> Result<(TrellisImageConditioningOutput, &'static str), String> {
    let runtime = TrellisImageConditioningRuntime::load_from_model_name(
        weights_root,
        image_large_root,
        model_name,
        prefer_wgpu,
    )?;
    let backend = runtime.backend_name();
    let output = runtime.extract_condition(preprocess, resolution)?;
    Ok((output, backend))
}

impl TrellisImageConditioningRuntime {
    pub fn load_from_model_name(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_name: &str,
        prefer_wgpu: bool,
    ) -> Result<Self, String> {
        #[cfg(feature = "runtime-model-wgpu")]
        if prefer_wgpu {
            let runtime = TrellisImageConditioningRuntimeImpl::<WgpuRuntimeBackend>::load_from_model_name(
                weights_root,
                image_large_root,
                model_name,
            )
            .map_err(|err| {
                format!(
                    "burn_trellis: failed to load TRELLIS image conditioning runtime on wgpu ({err}); refusing cpu fallback"
                )
            })?;
            return Ok(Self::Wgpu(Box::new(runtime)));
        }

        #[cfg(not(feature = "runtime-model-wgpu"))]
        if prefer_wgpu {
            return Err(
                "burn_trellis: TRELLIS image conditioning runtime requested wgpu but crate was built without runtime-model-wgpu"
                    .to_string(),
            );
        }

        let runtime =
            TrellisImageConditioningRuntimeImpl::<CpuRuntimeBackend>::load_from_model_name(
                weights_root,
                image_large_root,
                model_name,
            )?;
        Ok(Self::Cpu(Box::new(runtime)))
    }

    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(_) => "wgpu",
        }
    }

    pub fn extract_condition(
        &self,
        preprocess: &PreprocessOutput,
        resolution: usize,
    ) -> Result<TrellisImageConditioningOutput, String> {
        match self {
            Self::Cpu(runtime) => runtime.extract_condition(preprocess, resolution),
            #[cfg(feature = "runtime-model-wgpu")]
            Self::Wgpu(runtime) => runtime.extract_condition(preprocess, resolution),
        }
    }
}

#[derive(Debug)]
pub(crate) struct TrellisImageConditioningRuntimeImpl<B: Backend> {
    model: DinoVisionTransformer<B>,
    device: B::Device,
    patch_size: usize,
    register_token_count: usize,
    embedding_dim: usize,
}

impl<B: Backend> TrellisImageConditioningRuntimeImpl<B>
where
    B::Device: Default,
{
    fn load_from_model_name(
        weights_root: &Path,
        image_large_root: Option<&Path>,
        model_name: &str,
    ) -> Result<Self, String> {
        let assets = resolve_image_conditioning_assets(weights_root, image_large_root, model_name)?;
        let parsed_config = load_dinov3_config(assets.config_json.as_deref())?;
        let config = build_dinov3_config(parsed_config.as_ref());
        let source_label = assets.model_weights.path().display().to_string();
        let source_bytes = load_image_conditioning_weight_bytes(&assets.model_weights)?;

        let device = B::Device::default();
        let mut model = DinoVisionTransformer::<B>::new(&device, config.clone());
        let converted = convert_hf_dinov3(source_bytes.as_slice(), &config)?;
        let mut store = SafetensorsStore::from_bytes(Some(converted))
            .allow_partial(false)
            .validate(true);
        model.load_from(&mut store).map_err(|err| {
            format!(
                "failed to load DINOv3 image-conditioning weights '{}': {err}",
                source_label
            )
        })?;

        Ok(Self {
            model,
            device,
            patch_size: config.patch_size.max(1),
            register_token_count: config.register_token_count,
            embedding_dim: config.embedding_dimension,
        })
    }

    fn extract_condition(
        &self,
        preprocess: &PreprocessOutput,
        resolution: usize,
    ) -> Result<TrellisImageConditioningOutput, String> {
        if resolution == 0 {
            return Err("DINOv3 conditioning resolution must be > 0".to_string());
        }

        let input = preprocess_to_dino_tensor_values(preprocess, resolution)?;
        let tensor = Tensor::<B, 1>::from_floats(input.as_slice(), &self.device)
            .reshape([1, 3, resolution, resolution]);

        let (output, hooks, aux) =
            self.model
                .forward_with_intermediate_tokens_ext(tensor, &[], &[], None);
        // Hooks/aux are unused for runtime conditioning and can hold large
        // intermediate tensors; release them before downstream processing.
        drop(hooks);
        drop(aux);

        // TRELLIS python image_feature_extractor uses:
        //   hidden_states = model.embeddings(...)
        //   ... layer stack ...
        //   F.layer_norm(hidden_states, hidden_states.shape[-1:])
        // (without the model's learned final norm affine weights/bias).
        //
        // `burn_dino` returns both affine-normalized outputs and pre-final-norm
        // tokens (`x_prenorm`). Use the latter and apply plain layer norm to
        // match TRELLIS conditioning semantics.
        let prenorm = output.x_prenorm;
        let (var, mean) = prenorm.clone().var_mean_bias(2);
        let tokens: Tensor<B, 3> = prenorm.sub(mean).div(var.add_scalar(1.0e-5).sqrt());

        let dims = tokens.shape().dims::<3>();
        let patch_grid = resolution / self.patch_size.max(1);
        let expected_tokens = 1 + self.register_token_count + patch_grid.saturating_mul(patch_grid);
        if dims[1] != expected_tokens {
            return Err(format!(
                "DINOv3 conditioning token mismatch for resolution {}: got {}, expected {}",
                resolution, dims[1], expected_tokens
            ));
        }
        if dims[2] != self.embedding_dim {
            return Err(format!(
                "DINOv3 conditioning channel mismatch: got {}, expected {}",
                dims[2], self.embedding_dim
            ));
        }

        let values = tokens
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|err| format!("failed to read DINOv3 conditioning output: {err:?}"))?;

        Ok(TrellisImageConditioningOutput {
            resolution,
            token_count: dims[1],
            channels: dims[2],
            values,
        })
    }
}

#[derive(Debug)]
struct ImageConditioningAssets {
    model_weights: ImageConditioningWeights,
    config_json: Option<PathBuf>,
}

#[derive(Debug)]
enum ImageConditioningWeights {
    Safetensors(PathBuf),
    Burnpack(PathBuf),
}

impl ImageConditioningWeights {
    fn path(&self) -> &Path {
        match self {
            Self::Safetensors(path) | Self::Burnpack(path) => path.as_path(),
        }
    }
}

fn resolve_image_conditioning_assets(
    weights_root: &Path,
    image_large_root: Option<&Path>,
    model_name: &str,
) -> Result<ImageConditioningAssets, String> {
    let model_name = model_name.trim();
    if model_name.is_empty() {
        return Err(
            "Trellis pipeline image_cond_model.args.model_name is empty; cannot build runtime conditioning"
                .to_string(),
        );
    }

    let mut candidates = Vec::new();
    candidates.push(PathBuf::from(model_name));
    candidates.push(weights_root.join(model_name));
    if let Some(root) = image_large_root {
        candidates.push(root.join(model_name));
    }

    for candidate in candidates {
        if let Some(assets) = assets_from_candidate_path(candidate.as_path()) {
            return Ok(assets);
        }
    }

    if let Some(weights_path) = resolve_hf_hub_snapshot_weights(model_name) {
        let config_json = weights_path.parent().map(|dir| dir.join("config.json"));
        return Ok(ImageConditioningAssets {
            model_weights: ImageConditioningWeights::Safetensors(weights_path),
            config_json,
        });
    }

    let hf_roots = huggingface_hub_roots();
    let hf_roots_text = if hf_roots.is_empty() {
        "<none discovered>".to_string()
    } else {
        hf_roots
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    Err(format!(
        "failed to locate DINOv3 image-conditioning weights for model '{model_name}'. checked explicit/local paths under '{}' and '{}', and Hugging Face cache snapshots in: {}",
        weights_root.display(),
        image_large_root
            .map(|path| path.display().to_string())
            .unwrap_or_else(|| "<none>".to_string()),
        hf_roots_text
    ))
}

fn assets_from_candidate_path(candidate: &Path) -> Option<ImageConditioningAssets> {
    if !virtual_fs::exists(candidate) {
        return None;
    }

    if virtual_fs::is_file(candidate) {
        let extension = candidate.extension().and_then(|value| value.to_str())?;
        if extension.eq_ignore_ascii_case("safetensors") {
            return Some(ImageConditioningAssets {
                model_weights: ImageConditioningWeights::Safetensors(candidate.to_path_buf()),
                config_json: candidate.parent().map(|dir| dir.join("config.json")),
            });
        }
        if extension.eq_ignore_ascii_case("bpk") {
            return Some(ImageConditioningAssets {
                model_weights: ImageConditioningWeights::Burnpack(candidate.to_path_buf()),
                config_json: candidate.parent().map(|dir| dir.join("config.json")),
            });
        }
        return None;
    }

    let burnpack = candidate.join("model.bpk");
    if candidate_exists_or_has_parts(burnpack.as_path()) {
        return Some(ImageConditioningAssets {
            model_weights: ImageConditioningWeights::Burnpack(burnpack),
            config_json: Some(candidate.join("config.json")),
        });
    }

    let burnpack_f16 = candidate.join("model_f16.bpk");
    if candidate_exists_or_has_parts(burnpack_f16.as_path()) {
        return Some(ImageConditioningAssets {
            model_weights: ImageConditioningWeights::Burnpack(burnpack_f16),
            config_json: Some(candidate.join("config.json")),
        });
    }

    let weights = candidate.join("model.safetensors");
    if !virtual_fs::exists(weights.as_path()) {
        return None;
    }

    Some(ImageConditioningAssets {
        model_weights: ImageConditioningWeights::Safetensors(weights),
        config_json: Some(candidate.join("config.json")),
    })
}

#[cfg(target_arch = "wasm32")]
fn resolve_hf_hub_snapshot_weights(_model_name: &str) -> Option<PathBuf> {
    None
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_hf_hub_snapshot_weights(model_name: &str) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let suffix = model_name.rsplit('/').next().unwrap_or(model_name);

    for hub_root in huggingface_hub_roots() {
        let exact_repo = hub_root.join(format!("models--{}", model_name.replace('/', "--")));
        if let Some(path) = resolve_snapshot_weights_from_repo(exact_repo.as_path()) {
            let modified = path
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH);
            matches.push((modified, path));
        }

        let Ok(entries) = fs::read_dir(&hub_root) else {
            continue;
        };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let file_name = file_name.to_string_lossy();
            if !file_name.starts_with("models--") {
                continue;
            }
            if !file_name.ends_with(&format!("--{suffix}")) {
                continue;
            }
            if let Some(path) = resolve_snapshot_weights_from_repo(entry.path().as_path()) {
                let modified = path
                    .metadata()
                    .and_then(|metadata| metadata.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                matches.push((modified, path));
            }
        }
    }

    matches.sort_by_key(|entry| Reverse(entry.0));
    matches.into_iter().next().map(|(_, path)| path)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_snapshot_weights_from_repo(repo_root: &Path) -> Option<PathBuf> {
    if !repo_root.is_dir() {
        return None;
    }

    let refs_main = repo_root.join("refs/main");
    if let Ok(revision) = fs::read_to_string(refs_main.as_path()) {
        let revision = revision.trim();
        if !revision.is_empty() {
            let snapshot = repo_root.join("snapshots").join(revision);
            let weights = snapshot.join("model.safetensors");
            if weights.exists() {
                return Some(weights);
            }
        }
    }

    let snapshots = repo_root.join("snapshots");
    let mut candidates = Vec::new();
    let entries = fs::read_dir(snapshots).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let weights = path.join("model.safetensors");
        if !weights.exists() {
            continue;
        }
        let modified = weights
            .metadata()
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((modified, weights));
    }
    candidates.sort_by_key(|entry| Reverse(entry.0));
    candidates.into_iter().next().map(|(_, path)| path)
}

#[cfg(target_arch = "wasm32")]
fn huggingface_hub_roots() -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(not(target_arch = "wasm32"))]
fn huggingface_hub_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(path) = std::env::var_os("HF_HUB_CACHE") {
        push_unique_path(&mut roots, PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        push_unique_path(&mut roots, PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        push_unique_path(&mut roots, PathBuf::from(path).join("hub"));
    }
    if let Some(path) = std::env::var_os("HUGGINGFACE_HOME") {
        push_unique_path(&mut roots, PathBuf::from(path).join("hub"));
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        push_unique_path(
            &mut roots,
            PathBuf::from(path).join("huggingface").join("hub"),
        );
    }
    if let Some(path) = std::env::var_os("HOME") {
        push_unique_path(
            &mut roots,
            PathBuf::from(path)
                .join(".cache")
                .join("huggingface")
                .join("hub"),
        );
    }

    // Shared host caches are commonly mounted under /media or /mnt in multi-disk Linux setups.
    for base in ["/media", "/mnt"] {
        let Ok(entries) = fs::read_dir(base) else {
            continue;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join("models").join("huggingface").join("hub");
            push_unique_path(&mut roots, candidate);
        }
    }

    roots.into_iter().filter(|path| path.is_dir()).collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn push_unique_path(paths: &mut Vec<PathBuf>, candidate: PathBuf) {
    if !paths.iter().any(|existing| existing == &candidate) {
        paths.push(candidate);
    }
}

#[derive(Debug, Clone, Deserialize)]
struct HfDinoV3Config {
    hidden_size: Option<usize>,
    image_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_attention_heads: Option<usize>,
    num_channels: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_register_tokens: Option<usize>,
    patch_size: Option<usize>,
    proj_bias: Option<bool>,
    rope_theta: Option<f32>,
}

fn load_dinov3_config(path: Option<&Path>) -> Result<Option<HfDinoV3Config>, String> {
    let Some(path) = path else {
        return Ok(None);
    };
    if !virtual_fs::exists(path) {
        return Ok(None);
    }

    let bytes = virtual_fs::read(path)
        .map_err(|err| format!("failed to read DINOv3 config '{}': {err}", path.display()))?;
    let parsed: HfDinoV3Config = serde_json::from_slice(bytes.as_slice())
        .map_err(|err| format!("failed to parse DINOv3 config '{}': {err}", path.display()))?;
    Ok(Some(parsed))
}

fn build_dinov3_config(parsed: Option<&HfDinoV3Config>) -> DinoVisionTransformerConfig {
    let image_size = parsed.and_then(|cfg| cfg.image_size).unwrap_or(224).max(16);
    let patch_size = parsed.and_then(|cfg| cfg.patch_size).unwrap_or(16).max(1);
    let hidden_size = parsed
        .and_then(|cfg| cfg.hidden_size)
        .unwrap_or(1024)
        .max(1);
    let depth = parsed
        .and_then(|cfg| cfg.num_hidden_layers)
        .unwrap_or(24)
        .max(1);
    let num_heads = parsed
        .and_then(|cfg| cfg.num_attention_heads)
        .unwrap_or(16)
        .max(1);
    let register_tokens = parsed
        .and_then(|cfg| cfg.num_register_tokens)
        .unwrap_or(4)
        .max(0);
    let input_channels = parsed.and_then(|cfg| cfg.num_channels).unwrap_or(3).max(1);
    let rope_theta = parsed.and_then(|cfg| cfg.rope_theta).unwrap_or(100.0);
    let proj_bias = parsed.and_then(|cfg| cfg.proj_bias).unwrap_or(true);
    let mlp_ratio = parsed
        .and_then(|cfg| cfg.intermediate_size)
        .map(|intermediate| intermediate as f32 / hidden_size as f32)
        .filter(|ratio| ratio.is_finite() && *ratio > 0.0)
        .unwrap_or(4.0);

    let patch_grid = (image_size / patch_size).max(1);
    let num_patches = patch_grid.saturating_mul(patch_grid);
    let mut config = DinoVisionTransformerConfig::new(
        image_size,
        patch_size,
        input_channels,
        hidden_size,
        depth,
        burn_dino::layers::block::BlockConfig {
            attn: burn_dino::layers::attention::AttentionConfig {
                dim: hidden_size,
                num_heads,
                qkv_bias: true,
                proj_bias,
                ..Default::default()
            },
            layer_scale: Some(burn_dino::layers::layer_scale::LayerScaleConfig {
                dim: hidden_size,
            }),
            mlp_ratio,
        },
        nn::interpolate::Interpolate2dConfig {
            mode: nn::interpolate::InterpolateMode::Cubic,
            output_size: Some([patch_grid, patch_grid]),
            scale_factor: None,
            align_corners: true,
        },
        num_patches,
    );
    config.register_token_count = register_tokens;
    config.use_register_tokens = register_tokens > 0;
    config.use_mask_token = true;
    config.normalize_intermediate_tokens = true;
    config.rope_block_start = Some(0);
    config.rope_frequency = rope_theta;
    config.initializer = nn::Initializer::Zeros;
    config
}

fn preprocess_to_dino_tensor_values(
    preprocess: &PreprocessOutput,
    resolution: usize,
) -> Result<Vec<f32>, String> {
    if preprocess.width == 0 || preprocess.height == 0 {
        return Err("preprocess output is empty; cannot compute DINOv3 conditioning".to_string());
    }

    let expected = (preprocess.width as usize)
        .saturating_mul(preprocess.height as usize)
        .saturating_mul(3);
    if preprocess.rgb.len() != expected {
        return Err(format!(
            "preprocess rgb shape mismatch: width={} height={} expected={} actual={}",
            preprocess.width,
            preprocess.height,
            expected,
            preprocess.rgb.len()
        ));
    }

    let source =
        image::RgbImage::from_raw(preprocess.width, preprocess.height, preprocess.rgb.clone())
            .ok_or_else(|| {
                "failed to build source RGB image for DINOv3 conditioning".to_string()
            })?;
    let resized = image::imageops::resize(
        &source,
        resolution as u32,
        resolution as u32,
        // Match TRELLIS python image_feature_extractor.py (PIL.Image.LANCZOS)
        image::imageops::FilterType::Lanczos3,
    );

    let pixel_count = resolution.saturating_mul(resolution);
    let mut values = vec![0.0f32; pixel_count.saturating_mul(3)];
    for y in 0..resolution {
        for x in 0..resolution {
            let pixel = resized.get_pixel(x as u32, y as u32).0;
            let offset = y.saturating_mul(resolution).saturating_add(x);
            for channel in 0..3 {
                let scaled = pixel[channel] as f32 / 255.0;
                let normalized = (scaled - DINO_IMAGE_MEAN[channel]) / DINO_IMAGE_STD[channel];
                values[channel.saturating_mul(pixel_count).saturating_add(offset)] = normalized;
            }
        }
    }
    Ok(values)
}

#[derive(Default)]
struct QkvParts {
    q_weight: Option<Vec<f32>>,
    k_weight: Option<Vec<f32>>,
    v_weight: Option<Vec<f32>>,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    out_dim: Option<usize>,
    in_dim: Option<usize>,
}

#[derive(Debug)]
struct OwnedTensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<u8>,
}

impl OwnedTensor {
    fn from_f32(name: String, shape: Vec<usize>, values: &[f32]) -> Self {
        Self {
            name,
            shape,
            data: f32_values_to_bytes(values),
        }
    }
}

#[cfg(test)]
#[derive(Module, Debug)]
struct BinaryBlob<B: Backend> {
    bytes: Param<Tensor<B, 1, Int>>,
}

fn load_image_conditioning_weight_bytes(
    source: &ImageConditioningWeights,
) -> Result<Vec<u8>, String> {
    match source {
        ImageConditioningWeights::Safetensors(path) => virtual_fs::read(path).map_err(|err| {
            format!(
                "failed to read DINOv3 safetensors '{}': {err}",
                path.display()
            )
        }),
        ImageConditioningWeights::Burnpack(path) => {
            load_blob_bytes_from_burnpack_or_parts(path, load_burnpack_blob_bytes)
        }
    }
}

fn load_burnpack_blob_bytes(path: &Path) -> Result<Vec<u8>, String> {
    load_blob_bytes_from_blob_burnpack(path)
}

#[cfg(test)]
fn metadata_path(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("model.bpk");
    path.with_file_name(format!("{file_name}.meta.json"))
}

fn convert_hf_dinov3(
    bytes: &[u8],
    config: &DinoVisionTransformerConfig,
) -> Result<Vec<u8>, String> {
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|err| format!("failed to parse DINOv3 safetensors bytes: {err}"))?;

    let mut owned = Vec::<OwnedTensor>::new();
    let mut qkv_parts = BTreeMap::<usize, QkvParts>::new();
    let mut skipped = Vec::<String>::new();
    let mut saw_pos_embed = false;

    for name in tensors.names() {
        let view = tensors
            .tensor(name)
            .map_err(|err| format!("missing tensor '{name}' in DINOv3 safetensors: {err}"))?;
        if let Some(mapped) = map_direct_tensor(name, &view)? {
            if mapped.name == "pos_embed" {
                saw_pos_embed = true;
            }
            owned.push(mapped);
            continue;
        }

        if map_layer_tensor(name, &view, &mut owned, &mut qkv_parts)? {
            continue;
        }

        skipped.push(name.to_string());
    }

    for (layer, parts) in qkv_parts {
        let out_dim = parts
            .out_dim
            .ok_or_else(|| format!("missing qkv out_dim for DINOv3 layer {layer}"))?;
        let in_dim = parts
            .in_dim
            .ok_or_else(|| format!("missing qkv in_dim for DINOv3 layer {layer}"))?;
        let q = parts
            .q_weight
            .ok_or_else(|| format!("missing q_proj.weight for DINOv3 layer {layer}"))?;
        let k = parts
            .k_weight
            .ok_or_else(|| format!("missing k_proj.weight for DINOv3 layer {layer}"))?;
        let v = parts
            .v_weight
            .ok_or_else(|| format!("missing v_proj.weight for DINOv3 layer {layer}"))?;
        let expected_weight = out_dim.saturating_mul(in_dim);
        if q.len() != expected_weight || k.len() != expected_weight || v.len() != expected_weight {
            return Err(format!(
                "invalid qkv weight lengths in DINOv3 layer {layer}: q={} k={} v={} expected={}",
                q.len(),
                k.len(),
                v.len(),
                expected_weight
            ));
        }

        let mut qkv_weight = Vec::with_capacity(expected_weight * 3);
        qkv_weight.extend_from_slice(q.as_slice());
        qkv_weight.extend_from_slice(k.as_slice());
        qkv_weight.extend_from_slice(v.as_slice());
        let qkv_weight = transpose_2d_f32(qkv_weight.as_slice(), out_dim * 3, in_dim)?;
        owned.push(OwnedTensor::from_f32(
            format!("blocks.{layer}.attn.qkv.weight"),
            vec![in_dim, out_dim * 3],
            qkv_weight.as_slice(),
        ));

        let zero_bias = vec![0.0f32; out_dim];
        let q_bias = parts.q_bias.unwrap_or_else(|| zero_bias.clone());
        let k_bias = parts.k_bias.unwrap_or_else(|| zero_bias.clone());
        let v_bias = parts.v_bias.unwrap_or_else(|| zero_bias.clone());
        if q_bias.len() != out_dim || k_bias.len() != out_dim || v_bias.len() != out_dim {
            return Err(format!(
                "invalid qkv bias lengths in DINOv3 layer {layer}: q={} k={} v={} expected={}",
                q_bias.len(),
                k_bias.len(),
                v_bias.len(),
                out_dim
            ));
        }

        let mut qkv_bias = Vec::with_capacity(out_dim * 3);
        qkv_bias.extend_from_slice(q_bias.as_slice());
        qkv_bias.extend_from_slice(k_bias.as_slice());
        qkv_bias.extend_from_slice(v_bias.as_slice());
        owned.push(OwnedTensor::from_f32(
            format!("blocks.{layer}.attn.qkv.bias"),
            vec![out_dim * 3],
            qkv_bias.as_slice(),
        ));
    }

    if !saw_pos_embed {
        let grid = (config.image_size / config.patch_size.max(1)).max(1);
        let patch_tokens = grid.saturating_mul(grid);
        let token_count = 1 + config.register_token_count + patch_tokens;
        let values = vec![0.0f32; token_count.saturating_mul(config.embedding_dimension)];
        owned.push(OwnedTensor::from_f32(
            "pos_embed".to_string(),
            vec![1, token_count, config.embedding_dimension],
            values.as_slice(),
        ));
    }

    if !skipped.is_empty() {
        skipped.sort();
        let preview = skipped
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        let suffix = if skipped.len() > 12 {
            format!(", ... ({} total)", skipped.len())
        } else {
            format!(" ({} total)", skipped.len())
        };
        return Err(format!(
            "unsupported DINOv3 tensor keys encountered during conversion: {}{}",
            preview, suffix
        ));
    }

    let views = owned
        .iter()
        .map(|tensor| {
            TensorView::new(Dtype::F32, tensor.shape.clone(), tensor.data.as_slice())
                .map(|view| (tensor.name.clone(), view))
                .map_err(|err| format!("failed to build safetensors view '{}': {err}", tensor.name))
        })
        .collect::<Result<Vec<_>, _>>()?;

    serialize(views, None)
        .map_err(|err| format!("failed to serialize converted DINOv3 weights: {err}"))
}

fn map_direct_tensor(name: &str, view: &TensorView<'_>) -> Result<Option<OwnedTensor>, String> {
    let mapped = match name {
        "embeddings.cls_token" => Some("cls_token"),
        "embeddings.mask_token" => Some("mask_token"),
        "embeddings.register_tokens" => Some("register_tokens"),
        "embeddings.patch_embeddings.weight" => Some("patch_embed.proj.weight"),
        "embeddings.patch_embeddings.bias" => Some("patch_embed.proj.bias"),
        "embeddings.position_embeddings" => Some("pos_embed"),
        "norm.weight" => Some("norm.gamma"),
        "norm.bias" => Some("norm.beta"),
        _ => None,
    };

    let Some(mapped) = mapped else {
        return Ok(None);
    };

    let values = tensor_view_to_f32(view)?;
    let mut shape = view.shape().to_vec();
    if name == "embeddings.mask_token" && shape.len() == 3 && shape[0] == 1 && shape[1] == 1 {
        shape = vec![1, shape[2]];
    }

    Ok(Some(OwnedTensor::from_f32(
        mapped.to_string(),
        shape,
        values.as_slice(),
    )))
}

fn map_layer_tensor(
    name: &str,
    view: &TensorView<'_>,
    owned: &mut Vec<OwnedTensor>,
    qkv_parts: &mut BTreeMap<usize, QkvParts>,
) -> Result<bool, String> {
    let Some(rest) = name.strip_prefix("layer.") else {
        return Ok(false);
    };
    let Some((layer_str, tail)) = rest.split_once('.') else {
        return Ok(false);
    };
    let layer: usize = layer_str
        .parse()
        .map_err(|_| format!("invalid DINOv3 layer index in tensor key '{name}'"))?;

    match tail {
        "norm1.weight" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.norm1.gamma"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "norm1.bias" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.norm1.beta"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "norm2.weight" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.norm2.gamma"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "norm2.bias" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.norm2.beta"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "mlp.up_proj.weight" => {
            let (values, shape) = transpose_linear_weight(view, name)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.mlp.fc1.weight"),
                shape,
                values.as_slice(),
            ));
            Ok(true)
        }
        "mlp.up_proj.bias" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.mlp.fc1.bias"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "mlp.down_proj.weight" => {
            let (values, shape) = transpose_linear_weight(view, name)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.mlp.fc2.weight"),
                shape,
                values.as_slice(),
            ));
            Ok(true)
        }
        "mlp.down_proj.bias" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.mlp.fc2.bias"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "layer_scale1.lambda1" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.ls1.gamma"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "layer_scale2.lambda1" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.ls2.gamma"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "attention.o_proj.weight" => {
            let (values, shape) = transpose_linear_weight(view, name)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.attn.proj.weight"),
                shape,
                values.as_slice(),
            ));
            Ok(true)
        }
        "attention.o_proj.bias" => {
            let values = tensor_view_to_f32(view)?;
            owned.push(OwnedTensor::from_f32(
                format!("blocks.{layer}.attn.proj.bias"),
                view.shape().to_vec(),
                values.as_slice(),
            ));
            Ok(true)
        }
        "attention.q_proj.weight"
        | "attention.k_proj.weight"
        | "attention.v_proj.weight"
        | "attention.q_proj.bias"
        | "attention.k_proj.bias"
        | "attention.v_proj.bias" => {
            let entry = qkv_parts.entry(layer).or_default();
            assign_qkv_part(entry, name, view)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn assign_qkv_part(parts: &mut QkvParts, name: &str, view: &TensorView<'_>) -> Result<(), String> {
    let values = tensor_view_to_f32(view)?;
    if name.ends_with(".weight") {
        if view.shape().len() != 2 {
            return Err(format!(
                "invalid qkv weight shape for '{name}': expected rank-2, got {:?}",
                view.shape()
            ));
        }
        let out_dim = view.shape()[0];
        let in_dim = view.shape()[1];
        if let Some(existing) = parts.out_dim
            && existing != out_dim
        {
            return Err(format!(
                "qkv out_dim mismatch for '{name}': {} vs {}",
                existing, out_dim
            ));
        }
        if let Some(existing) = parts.in_dim
            && existing != in_dim
        {
            return Err(format!(
                "qkv in_dim mismatch for '{name}': {} vs {}",
                existing, in_dim
            ));
        }
        parts.out_dim = Some(out_dim);
        parts.in_dim = Some(in_dim);
    }

    if name.ends_with("attention.q_proj.weight") {
        parts.q_weight = Some(values);
    } else if name.ends_with("attention.k_proj.weight") {
        parts.k_weight = Some(values);
    } else if name.ends_with("attention.v_proj.weight") {
        parts.v_weight = Some(values);
    } else if name.ends_with("attention.q_proj.bias") {
        parts.q_bias = Some(values);
    } else if name.ends_with("attention.k_proj.bias") {
        parts.k_bias = Some(values);
    } else if name.ends_with("attention.v_proj.bias") {
        parts.v_bias = Some(values);
    }

    Ok(())
}

fn transpose_linear_weight(
    view: &TensorView<'_>,
    name: &str,
) -> Result<(Vec<f32>, Vec<usize>), String> {
    if view.shape().len() != 2 {
        return Err(format!(
            "invalid linear weight shape for '{name}': expected rank-2, got {:?}",
            view.shape()
        ));
    }
    let out_dim = view.shape()[0];
    let in_dim = view.shape()[1];
    let values = tensor_view_to_f32(view)?;
    let transposed = transpose_2d_f32(values.as_slice(), out_dim, in_dim)?;
    Ok((transposed, vec![in_dim, out_dim]))
}

fn transpose_2d_f32(values: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, String> {
    let expected = rows.saturating_mul(cols);
    if values.len() != expected {
        return Err(format!(
            "cannot transpose matrix with shape [{rows}, {cols}]: {} values provided",
            values.len()
        ));
    }
    let mut out = vec![0.0f32; expected];
    for row in 0..rows {
        let row_offset = row * cols;
        for col in 0..cols {
            out[col * rows + row] = values[row_offset + col];
        }
    }
    Ok(out)
}

fn tensor_view_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>, String> {
    match view.dtype() {
        Dtype::F32 => f32_bytes_to_vec(view.data()),
        Dtype::F16 => f16_bytes_to_vec(view.data()),
        Dtype::BF16 => bf16_bytes_to_vec(view.data()),
        other => Err(format!("unsupported DINOv3 tensor dtype: {other:?}")),
    }
}

fn f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err(format!(
            "invalid f32 byte length {}; not divisible by 4",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn f16_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid f16 byte length {}; not divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(half::f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn bf16_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(format!(
            "invalid bf16 byte length {}; not divisible by 2",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(half::bf16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn f32_values_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(value.to_le_bytes().as_slice());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::{
        BinaryBlob, ImageConditioningWeights, assets_from_candidate_path, build_dinov3_config,
        f16_bytes_to_vec, huggingface_hub_roots, load_image_conditioning_weight_bytes,
        metadata_path, transpose_2d_f32,
    };
    use burn::module::{Param, ParamId};
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_store::{BurnpackStore, ModuleSnapshot};

    fn write_blob_burnpack(path: &std::path::Path, bytes: &[u8]) {
        type BlobBackend = burn::backend::NdArray<f32, u8>;
        let device = <BlobBackend as burn::tensor::backend::BackendTypes>::Device::default();
        let tensor = Tensor::<BlobBackend, 1, Int>::from_data(
            TensorData::new(bytes.to_vec(), [bytes.len()]),
            &device,
        );
        let blob = BinaryBlob {
            bytes: Param::initialized(ParamId::new(), tensor),
        };
        let mut store = BurnpackStore::from_file(path).overwrite(true);
        blob.save_into(&mut store).expect("save burnpack blob");
        std::fs::write(
            metadata_path(path),
            serde_json::to_vec_pretty(&serde_json::json!({ "bytes_len": bytes.len() }))
                .expect("serialize metadata"),
        )
        .expect("write burnpack metadata");
    }

    #[test]
    fn dino_config_defaults_to_register_tokens() {
        let config = build_dinov3_config(None);
        assert_eq!(config.register_token_count, 4);
        assert!(config.use_register_tokens);
        assert_eq!(config.rope_block_start, Some(0));
    }

    #[test]
    fn f16_decode_roundtrip_values() {
        let bytes = [0x00u8, 0x3c, 0x00, 0xbc];
        let values = f16_bytes_to_vec(bytes.as_slice()).expect("decode f16");
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 1.0e-6);
        assert!((values[1] + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn assets_resolve_for_local_snapshot_dir() {
        let root = tempfile::tempdir().expect("temp dir");
        let snapshot = root.path().join("snapshot");
        std::fs::create_dir_all(&snapshot).expect("create snapshot");
        std::fs::write(snapshot.join("model.safetensors"), b"dummy").expect("write weights");
        std::fs::write(snapshot.join("config.json"), b"{}").expect("write config");
        let assets = assets_from_candidate_path(snapshot.as_path()).expect("assets from snapshot");
        assert!(matches!(
            assets.model_weights,
            ImageConditioningWeights::Safetensors(ref path) if path.ends_with("model.safetensors")
        ));
        assert!(
            assets
                .config_json
                .as_ref()
                .is_some_and(|path| path.ends_with("config.json"))
        );
    }

    #[test]
    fn assets_resolve_and_load_from_burnpack_parts() {
        let root = tempfile::tempdir().expect("temp dir");
        let model_dir = root.path().join("facebook").join("dinov3-test");
        std::fs::create_dir_all(&model_dir).expect("create model dir");
        std::fs::write(model_dir.join("config.json"), b"{}").expect("write config");

        let payload = b"dinov3_payload_bytes";
        let burnpack_path = model_dir.join("model_f16.bpk");
        write_blob_burnpack(&burnpack_path, payload);

        let part_path = model_dir.join("model_f16.bpk.part-00000.bpk");
        std::fs::rename(&burnpack_path, &part_path).expect("move burnpack into part");
        std::fs::rename(metadata_path(&burnpack_path), metadata_path(&part_path))
            .expect("move part metadata");
        let part_bytes = std::fs::metadata(&part_path).expect("part metadata").len() as usize;

        let manifest_path = model_dir.join("model_f16.bpk.parts.json");
        let part_name = part_path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("part file name");
        let manifest = serde_json::json!({
            "version": 1,
            "source_file": "model_f16.bpk",
            "source_modified_unix_ms": 0,
            "total_bytes": part_bytes,
            "max_part_bytes": part_bytes,
            "parts": [
                {
                    "path": part_name,
                    "bytes": part_bytes,
                    "sha256": "",
                    "tensors": 1
                }
            ]
        });
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("write manifest");

        let assets = assets_from_candidate_path(model_dir.as_path()).expect("resolve assets");
        assert!(matches!(
            assets.model_weights,
            ImageConditioningWeights::Burnpack(ref path) if path.ends_with("model_f16.bpk")
        ));
        let loaded = load_image_conditioning_weight_bytes(&assets.model_weights)
            .expect("load bytes from parts");
        assert_eq!(loaded, payload);
    }

    #[test]
    fn huggingface_roots_use_cache_convention() {
        for path in huggingface_hub_roots() {
            let as_text = path.display().to_string();
            assert!(as_text.contains("huggingface"));
            assert!(as_text.ends_with("/hub") || as_text.ends_with("\\hub"));
        }
    }

    #[test]
    fn transpose_2d_swaps_axes_row_major() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let transposed = transpose_2d_f32(src.as_slice(), 2, 3).expect("transpose");
        assert_eq!(transposed, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }
}

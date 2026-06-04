use std::path::{Path, PathBuf};

use burn::{
    module::{Module, ModuleMapper, Param},
    prelude::*,
    tensor::{Bytes, FloatDType},
};
use burn_store::{
    ApplyResult, BurnpackStore, KeyRemapper, ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore,
};
use burn_synth_import::parts::load_model_from_burnpack_parts;

use crate::{
    ElasticGaussianFixedlenDecoderConfig, LatentSeqMmFlowModel, LatentSeqMmFlowModelConfig,
    OctreeGaussianDecoder, OctreeProbabilityFixedlenDecoderConfig, TripoSplatArtifact,
    TripoSplatArtifactSet, TripoSplatBurnpackPrecision, TripoSplatRuntimeComponents,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripoSplatRuntimeLoadPhase {
    Loaded,
    Cast,
}

impl TripoSplatRuntimeLoadPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loaded => "loaded",
            Self::Cast => "cast",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripoSplatRuntimeLoadEvent {
    pub component: &'static str,
    pub phase: TripoSplatRuntimeLoadPhase,
}

impl TripoSplatRuntimeLoadEvent {
    pub fn label(self) -> String {
        format!("{}_{}", self.component, self.phase.as_str())
    }
}

pub fn load_triposplat_runtime_components<B: Backend>(
    device: &B::Device,
    artifacts: &TripoSplatArtifactSet,
) -> Result<TripoSplatRuntimeComponents<B>, Box<dyn std::error::Error>> {
    let compute_dtype = if artifacts.precision == TripoSplatBurnpackPrecision::F16 {
        // Official TripoSplat flow/decoder weights are distributed as fp16, but WGPU fp16
        // execution currently produces non-finite Gaussian features. Keep f16 storage/sharding
        // supported, then promote compute for numerically stable default inference.
        Some(FloatDType::F32)
    } else {
        None
    };
    load_triposplat_runtime_components_with_compute_dtype(device, artifacts, compute_dtype)
}

pub fn load_triposplat_runtime_components_with_compute_dtype<B: Backend>(
    device: &B::Device,
    artifacts: &TripoSplatArtifactSet,
    compute_dtype: Option<FloatDType>,
) -> Result<TripoSplatRuntimeComponents<B>, Box<dyn std::error::Error>> {
    load_triposplat_runtime_components_with_compute_dtype_and_callback(
        device,
        artifacts,
        compute_dtype,
        |_| Ok::<(), Box<dyn std::error::Error>>(()),
    )
}

pub fn load_triposplat_runtime_components_with_compute_dtype_and_callback<B, F>(
    device: &B::Device,
    artifacts: &TripoSplatArtifactSet,
    compute_dtype: Option<FloatDType>,
    mut after_component: F,
) -> Result<TripoSplatRuntimeComponents<B>, Box<dyn std::error::Error>>
where
    B: Backend,
    F: FnMut(TripoSplatRuntimeLoadEvent) -> Result<(), Box<dyn std::error::Error>>,
{
    let dino = required_artifact("dino_v3_vit_h")?;
    let flux = required_artifact("flux2_vae_encoder")?;
    let flow = required_artifact("triposplat_flow")?;
    let decoder = required_artifact("triposplat_vae_decoder")?;

    let mut dinov3 = load_dinov3_artifact(device, dino, artifacts)?;
    after_component(load_event(
        "dino_v3_vit_h",
        TripoSplatRuntimeLoadPhase::Loaded,
    ))?;
    if should_cast_artifact(artifacts.precision, compute_dtype) {
        dinov3 = cast_module_float_dtype(dinov3, compute_dtype.expect("checked cast dtype"));
        after_component(load_event(
            "dino_v3_vit_h",
            TripoSplatRuntimeLoadPhase::Cast,
        ))?;
    }

    let mut flux2_vae_encoder = load_flux2_artifact(device, flux, artifacts)?;
    after_component(load_event(
        "flux2_vae_encoder",
        TripoSplatRuntimeLoadPhase::Loaded,
    ))?;
    if should_cast_artifact(artifacts.precision, compute_dtype) {
        flux2_vae_encoder = cast_module_float_dtype(
            flux2_vae_encoder,
            compute_dtype.expect("checked cast dtype"),
        );
        after_component(load_event(
            "flux2_vae_encoder",
            TripoSplatRuntimeLoadPhase::Cast,
        ))?;
    }

    let mut flow = load_flow_artifact(device, flow, artifacts)?;
    after_component(load_event(
        "triposplat_flow",
        TripoSplatRuntimeLoadPhase::Loaded,
    ))?;
    if should_cast_artifact(artifacts.precision, compute_dtype) {
        flow = cast_module_float_dtype(flow, compute_dtype.expect("checked cast dtype"));
        after_component(load_event(
            "triposplat_flow",
            TripoSplatRuntimeLoadPhase::Cast,
        ))?;
    }

    let mut decoder = load_decoder_artifact(device, decoder, artifacts)?;
    after_component(load_event(
        "triposplat_vae_decoder",
        TripoSplatRuntimeLoadPhase::Loaded,
    ))?;
    if should_cast_artifact(artifacts.precision, compute_dtype) {
        decoder = cast_module_float_dtype(decoder, compute_dtype.expect("checked cast dtype"));
        after_component(load_event(
            "triposplat_vae_decoder",
            TripoSplatRuntimeLoadPhase::Cast,
        ))?;
    }

    Ok(TripoSplatRuntimeComponents {
        dinov3,
        flux2_vae_encoder,
        flow,
        decoder,
    })
}

pub fn load_triposplat_runtime_components_from_root<B: Backend>(
    device: &B::Device,
    root: impl Into<PathBuf>,
    precision: TripoSplatBurnpackPrecision,
) -> Result<TripoSplatRuntimeComponents<B>, Box<dyn std::error::Error>> {
    let artifacts = TripoSplatArtifactSet::new(root, precision);
    load_triposplat_runtime_components(device, &artifacts)
}

pub fn load_triposplat_runtime_components_from_root_with_compute_dtype<B: Backend>(
    device: &B::Device,
    root: impl Into<PathBuf>,
    precision: TripoSplatBurnpackPrecision,
    compute_dtype: Option<FloatDType>,
) -> Result<TripoSplatRuntimeComponents<B>, Box<dyn std::error::Error>> {
    let artifacts = TripoSplatArtifactSet::new(root, precision);
    load_triposplat_runtime_components_with_compute_dtype(device, &artifacts, compute_dtype)
}

pub fn load_triposplat_flow_from_safetensors<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    config: &LatentSeqMmFlowModelConfig,
) -> Result<LatentSeqMmFlowModel<B>, Box<dyn std::error::Error>> {
    let mut model = config.clone().init(device);
    let mut store = build_flow_store(path.as_ref())?;
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("failed to load TripoSplat flow weights: {err}"))?;
    validate_nonempty_apply("TripoSplat flow safetensors", &result)?;
    model.reset_canonical_pos_pe(device);
    Ok(model)
}

pub fn load_triposplat_flow_from_burnpack_file<B: Backend>(
    device: &B::Device,
    burnpack_path: impl AsRef<Path>,
    config: &LatentSeqMmFlowModelConfig,
) -> Result<LatentSeqMmFlowModel<B>, Box<dyn std::error::Error>> {
    let mut model = config.clone().init(device);
    let mut store =
        BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
    model
        .load_from(&mut store)
        .map_err(|err| format!("failed to load TripoSplat flow burnpack: {err}"))?;
    model.reset_canonical_pos_pe(device);
    Ok(model)
}

pub fn apply_triposplat_flow_burnpack_part_bytes<B: Backend>(
    model: &mut LatentSeqMmFlowModel<B>,
    burnpack_bytes: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
        .allow_partial(true)
        .validate(should_validate_burnpack());
    model
        .load_from(&mut store)
        .map_err(|err| format!("failed to apply TripoSplat flow burnpack part: {err}"))?;
    Ok(())
}

pub fn import_triposplat_flow_burnpack_to_path<B: Backend>(
    device: &B::Device,
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    config: &LatentSeqMmFlowModelConfig,
    use_f16: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut model = load_triposplat_flow_from_safetensors::<B>(device, source_path, config)?;
    let dtype = if use_f16 {
        FloatDType::F16
    } else {
        FloatDType::F32
    };
    model = cast_module_float_dtype(model, dtype);
    save_burnpack(&model, output_path.as_ref(), "TripoSplat flow")?;
    Ok(output_path.as_ref().to_path_buf())
}

pub fn load_triposplat_decoder_from_safetensors<B: Backend>(
    device: &B::Device,
    path: impl AsRef<Path>,
    octree_config: &OctreeProbabilityFixedlenDecoderConfig,
    gs_config: &ElasticGaussianFixedlenDecoderConfig,
) -> Result<OctreeGaussianDecoder<B>, Box<dyn std::error::Error>> {
    let mut model = OctreeGaussianDecoder::new(device, octree_config.clone(), gs_config.clone());
    let mut store = build_decoder_store(path.as_ref())?;
    let result = model
        .load_from(&mut store)
        .map_err(|err| format!("failed to load TripoSplat decoder weights: {err}"))?;
    validate_nonempty_apply("TripoSplat decoder safetensors", &result)?;
    Ok(model)
}

pub fn load_triposplat_decoder_from_burnpack_file<B: Backend>(
    device: &B::Device,
    burnpack_path: impl AsRef<Path>,
    octree_config: &OctreeProbabilityFixedlenDecoderConfig,
    gs_config: &ElasticGaussianFixedlenDecoderConfig,
) -> Result<OctreeGaussianDecoder<B>, Box<dyn std::error::Error>> {
    let mut model = OctreeGaussianDecoder::new(device, octree_config.clone(), gs_config.clone());
    let mut store =
        BurnpackStore::from_file(burnpack_path.as_ref()).validate(should_validate_burnpack());
    model
        .load_from(&mut store)
        .map_err(|err| format!("failed to load TripoSplat decoder burnpack: {err}"))?;
    Ok(model)
}

pub fn apply_triposplat_decoder_burnpack_part_bytes<B: Backend>(
    model: &mut OctreeGaussianDecoder<B>,
    burnpack_bytes: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(burnpack_bytes)))
        .allow_partial(true)
        .validate(should_validate_burnpack());
    model
        .load_from(&mut store)
        .map_err(|err| format!("failed to apply TripoSplat decoder burnpack part: {err}"))?;
    Ok(())
}

pub fn import_triposplat_decoder_burnpack_to_path<B: Backend>(
    device: &B::Device,
    source_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    octree_config: &OctreeProbabilityFixedlenDecoderConfig,
    gs_config: &ElasticGaussianFixedlenDecoderConfig,
    use_f16: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let mut model = load_triposplat_decoder_from_safetensors::<B>(
        device,
        source_path,
        octree_config,
        gs_config,
    )?;
    let dtype = if use_f16 {
        FloatDType::F16
    } else {
        FloatDType::F32
    };
    model = cast_module_float_dtype(model, dtype);
    save_burnpack(&model, output_path.as_ref(), "TripoSplat decoder")?;
    Ok(output_path.as_ref().to_path_buf())
}

pub fn triposplat_common_key_remap_rules() -> &'static [(&'static str, &'static str)] {
    &[
        (
            r"^(.*)_embedder\.mlp\.0\.(weight|bias)$",
            "${1}_embedder.mlp_0.$2",
        ),
        (
            r"^(.*)_embedder\.mlp\.2\.(weight|bias)$",
            "${1}_embedder.mlp_2.$2",
        ),
        (
            r"^(.*)adaLN_modulation\.1\.(weight|bias)$",
            "${1}ada_ln_modulation.$2",
        ),
        (r"^(.*\.adaLN_modulation)\.1\.(weight|bias)$", "$1.$2"),
        (r"^(.*\.mlp)\.mlp\.0\.(weight|bias)$", "$1.mlp_0.$2"),
        (r"^(.*\.mlp)\.mlp\.2\.(weight|bias)$", "$1.mlp_2.$2"),
        (
            r"^cam_refiner\.mlp\.0\.(weight|bias)$",
            "cam_refiner.layers.0.$1",
        ),
        (
            r"^cam_refiner\.mlp\.2\.(weight|bias)$",
            "cam_refiner.layers.1.$1",
        ),
        (r"^(.*)\.to_qkv\.(weight|bias)$", "$1.qkv.$2"),
        (r"^(.*)\.to_q\.(weight|bias)$", "$1.q.$2"),
        (r"^(.*)\.to_kv\.(weight|bias)$", "$1.kv.$2"),
        (r"^(.*)\.to_out\.(weight|bias)$", "$1.out.$2"),
        (r"^(.*)\.q_rms_norm\.gamma$", "$1.q_norm.gamma"),
        (r"^(.*)\.k_rms_norm\.gamma$", "$1.k_norm.gamma"),
        (r"^(.*)\.norm([123]?)\.weight$", "$1.norm$2.gamma"),
        (r"^(.*)\.norm([123]?)\.bias$", "$1.norm$2.beta"),
    ]
}

fn build_flow_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .remap(build_common_remapper()?)
        .validate(true))
}

fn build_decoder_store(path: &Path) -> Result<SafetensorsStore, Box<dyn std::error::Error>> {
    let mut remapper = build_common_remapper()?;
    add_gaussian_block_remaps(&mut remapper)?;
    Ok(SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter)
        .allow_partial(true)
        .remap(remapper)
        .validate(true))
}

fn build_common_remapper() -> Result<KeyRemapper, Box<dyn std::error::Error>> {
    let mut remapper = KeyRemapper::new();
    for &(from, to) in triposplat_common_key_remap_rules() {
        remapper = remapper
            .add_pattern(from, to)
            .map_err(|err| format!("invalid TripoSplat remap rule {from}->{to}: {err}"))?;
    }
    Ok(remapper)
}

fn add_gaussian_block_remaps(remapper: &mut KeyRemapper) -> Result<(), Box<dyn std::error::Error>> {
    for block in 0..64usize {
        let base = format!("^gs\\.blocks\\.{block}\\.");
        *remapper = std::mem::take(remapper)
            .add_pattern(
                format!("{base}self_attn\\.(.+)$"),
                format!("gs.self_attns.{block}.$1"),
            )
            .map_err(|err| format!("invalid Gaussian self-attn remap for block {block}: {err}"))?
            .add_pattern(
                format!("{base}cross_attn\\.(.+)$"),
                format!("gs.cross_attns.{block}.$1"),
            )
            .map_err(|err| format!("invalid Gaussian cross-attn remap for block {block}: {err}"))?
            .add_pattern(format!("{base}mlp\\.(.+)$"), format!("gs.mlps.{block}.$1"))
            .map_err(|err| format!("invalid Gaussian mlp remap for block {block}: {err}"))?
            .add_pattern(
                format!("{base}norm1\\.(weight|bias|gamma|beta)$"),
                format!("gs.norms.{}.${{1}}", block * 3),
            )
            .map_err(|err| format!("invalid Gaussian norm1 remap for block {block}: {err}"))?
            .add_pattern(
                format!("{base}norm2\\.(weight|bias|gamma|beta)$"),
                format!("gs.norms.{}.${{1}}", block * 3 + 1),
            )
            .map_err(|err| format!("invalid Gaussian norm2 remap for block {block}: {err}"))?
            .add_pattern(
                format!("{base}norm3\\.(weight|bias|gamma|beta)$"),
                format!("gs.norms.{}.${{1}}", block * 3 + 2),
            )
            .map_err(|err| format!("invalid Gaussian norm3 remap for block {block}: {err}"))?;
    }
    *remapper = std::mem::take(remapper)
        .add_pattern(r"^(gs\.norms\.\d+)\.weight$", "$1.gamma")
        .map_err(|err| format!("invalid Gaussian norm weight remap: {err}"))?
        .add_pattern(r"^(gs\.norms\.\d+)\.bias$", "$1.beta")
        .map_err(|err| format!("invalid Gaussian norm bias remap: {err}"))?;
    Ok(())
}

fn validate_nonempty_apply(
    label: &str,
    result: &ApplyResult,
) -> Result<(), Box<dyn std::error::Error>> {
    if result.applied.is_empty() {
        return Err(format!("{label} import did not apply any tensors").into());
    }
    Ok(())
}

fn required_artifact(stem: &str) -> Result<TripoSplatArtifact, Box<dyn std::error::Error>> {
    crate::artifact::TRIPOSPLAT_ARTIFACTS
        .into_iter()
        .find(|artifact| artifact.burnpack_stem == stem)
        .ok_or_else(|| format!("missing TripoSplat artifact metadata for {stem}").into())
}

fn load_event(
    component: &'static str,
    phase: TripoSplatRuntimeLoadPhase,
) -> TripoSplatRuntimeLoadEvent {
    TripoSplatRuntimeLoadEvent { component, phase }
}

fn should_cast_artifact(
    precision: TripoSplatBurnpackPrecision,
    compute_dtype: Option<FloatDType>,
) -> bool {
    !matches!(
        (precision, compute_dtype),
        (_, None)
            | (TripoSplatBurnpackPrecision::F32, Some(FloatDType::F32))
            | (TripoSplatBurnpackPrecision::F16, Some(FloatDType::F16))
    )
}

fn load_dinov3_artifact<B: Backend>(
    device: &B::Device,
    artifact: TripoSplatArtifact,
    artifacts: &TripoSplatArtifactSet,
) -> Result<burn_dino::model::dinov3::DinoV3ViT<B>, Box<dyn std::error::Error>> {
    let config = burn_dino::model::dinov3::DinoV3Config::vit_h_16_plus(None);
    let burnpack = artifact.burnpack_path(&artifacts.root, artifacts.precision);
    if burnpack.exists() {
        return burn_dino::model::dinov3::import::load_dinov3_from_burnpack_file(
            device, &burnpack, &config,
        );
    }
    load_model_from_burnpack_parts(
        std::slice::from_ref(&burnpack),
        "TripoSplat DINOv3",
        should_validate_burnpack(),
        || config.clone().init(device),
        |model, bytes| {
            burn_dino::model::dinov3::import::apply_dinov3_burnpack_part_bytes(model, bytes)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(|err| err.into())
    .and_then(|loaded| {
        loaded.ok_or_else(|| {
            format!(
                "missing TripoSplat DINOv3 burnpack or parts manifest at {}",
                burnpack.display()
            )
            .into()
        })
    })
}

fn load_flux2_artifact<B: Backend>(
    device: &B::Device,
    artifact: TripoSplatArtifact,
    artifacts: &TripoSplatArtifactSet,
) -> Result<burn_flux::Flux2VaeEncoder<B>, Box<dyn std::error::Error>> {
    let config = burn_flux::Flux2VaeEncoderConfig::flux2();
    let burnpack = artifact.burnpack_path(&artifacts.root, artifacts.precision);
    if burnpack.exists() {
        return burn_flux::flux2_import::load_flux2_vae_encoder_from_burnpack_file(
            device, &burnpack, &config,
        );
    }
    load_model_from_burnpack_parts(
        std::slice::from_ref(&burnpack),
        "TripoSplat Flux2 VAE encoder",
        should_validate_burnpack(),
        || config.clone().init(device),
        |model, bytes| {
            burn_flux::flux2_import::apply_flux2_vae_encoder_burnpack_part_bytes(model, bytes)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(|err| err.into())
    .and_then(|loaded| {
        loaded.ok_or_else(|| {
            format!(
                "missing TripoSplat Flux2 VAE burnpack or parts manifest at {}",
                burnpack.display()
            )
            .into()
        })
    })
}

fn load_flow_artifact<B: Backend>(
    device: &B::Device,
    artifact: TripoSplatArtifact,
    artifacts: &TripoSplatArtifactSet,
) -> Result<LatentSeqMmFlowModel<B>, Box<dyn std::error::Error>> {
    let config = LatentSeqMmFlowModelConfig::triposplat();
    let burnpack = artifact.burnpack_path(&artifacts.root, artifacts.precision);
    if burnpack.exists() {
        return load_triposplat_flow_from_burnpack_file(device, &burnpack, &config);
    }
    load_model_from_burnpack_parts(
        std::slice::from_ref(&burnpack),
        "TripoSplat flow",
        should_validate_burnpack(),
        || config.clone().init(device),
        |model, bytes| {
            apply_triposplat_flow_burnpack_part_bytes(model, bytes).map_err(|err| err.to_string())
        },
    )
    .map_err(|err| err.into())
    .and_then(|loaded| {
        let mut loaded = loaded.ok_or_else(|| {
            format!(
                "missing TripoSplat flow burnpack or parts manifest at {}",
                burnpack.display()
            )
        })?;
        loaded.reset_canonical_pos_pe(device);
        Ok(loaded)
    })
}

fn load_decoder_artifact<B: Backend>(
    device: &B::Device,
    artifact: TripoSplatArtifact,
    artifacts: &TripoSplatArtifactSet,
) -> Result<OctreeGaussianDecoder<B>, Box<dyn std::error::Error>> {
    let octree_config = OctreeProbabilityFixedlenDecoderConfig::triposplat();
    let gs_config = ElasticGaussianFixedlenDecoderConfig::triposplat();
    let burnpack = artifact.burnpack_path(&artifacts.root, artifacts.precision);
    if burnpack.exists() {
        return load_triposplat_decoder_from_burnpack_file(
            device,
            &burnpack,
            &octree_config,
            &gs_config,
        );
    }
    load_model_from_burnpack_parts(
        std::slice::from_ref(&burnpack),
        "TripoSplat decoder",
        should_validate_burnpack(),
        || OctreeGaussianDecoder::new(device, octree_config.clone(), gs_config.clone()),
        |model, bytes| {
            apply_triposplat_decoder_burnpack_part_bytes(model, bytes)
                .map_err(|err| err.to_string())
        },
    )
    .map_err(|err| err.into())
    .and_then(|loaded| {
        loaded.ok_or_else(|| {
            format!(
                "missing TripoSplat decoder burnpack or parts manifest at {}",
                burnpack.display()
            )
            .into()
        })
    })
}

struct FloatDTypeMapper {
    dtype: FloatDType,
}

impl<B: Backend> ModuleMapper<B> for FloatDTypeMapper {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        Param::from_mapped_value(id, tensor.cast(self.dtype), mapper)
    }
}

fn cast_module_float_dtype<B: Backend, M: Module<B>>(module: M, dtype: FloatDType) -> M {
    let mut mapper = FloatDTypeMapper { dtype };
    module.map(&mut mapper)
}

fn save_burnpack<B: Backend, M: Module<B>>(
    model: &M,
    path: &Path,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut store = BurnpackStore::from_file(path).overwrite(true);
    model
        .save_into(&mut store)
        .map_err(|err| format!("failed to save {label} burnpack: {err}"))?;
    Ok(())
}

fn should_validate_burnpack() -> bool {
    cfg!(all(not(target_arch = "wasm32"), debug_assertions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FlowState, OctreeProbabilityFixedlenDecoder, TripoSplatCondition};
    use burn_synth_import::parts::write_burnpack_parts_for_wasm;

    type TestBackend = burn::backend::NdArray<f32>;

    fn unique_temp_dir(label: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "burn_triposplat_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    fn tensor_vec<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect("tensor vec")
    }

    fn assert_close(lhs: &[f32], rhs: &[f32], label: &str) {
        assert_eq!(lhs.len(), rhs.len(), "{label} length mismatch");
        let max_abs = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs <= 1.0e-6,
            "{label} max abs diff {max_abs} exceeds tolerance"
        );
    }

    fn remap_common(key: &str) -> String {
        let mut remapper = build_common_remapper().unwrap();
        add_gaussian_block_remaps(&mut remapper).unwrap();
        let mut out = key.to_string();
        for (pattern, replacement) in &remapper.patterns {
            if pattern.is_match(&out) {
                out = pattern.replace_all(&out, replacement.as_str()).to_string();
            }
        }
        out
    }

    #[test]
    fn remaps_flow_timestep_and_shared_adaln_keys() {
        assert_eq!(
            remap_common("t_embedder.mlp.0.weight"),
            "t_embedder.mlp_0.weight"
        );
        assert_eq!(
            remap_common("adaLN_modulation.1.bias"),
            "ada_ln_modulation.bias"
        );
    }

    #[test]
    fn remaps_gaussian_decoder_flattened_blocks() {
        assert_eq!(
            remap_common("gs.blocks.4.self_attn.to_qkv.weight"),
            "gs.self_attns.4.qkv.weight"
        );
        assert_eq!(
            remap_common("gs.blocks.4.norm2.weight"),
            "gs.norms.13.gamma"
        );
        assert_eq!(
            remap_common("octree.blocks.2.cross_attn.to_kv.weight"),
            "octree.blocks.2.cross_attn.kv.weight"
        );
    }

    #[test]
    fn flow_single_burnpack_and_parts_loads_are_numerically_equivalent() {
        let root = unique_temp_dir("flow_parts_parity");
        let bpk = root.join("triposplat_flow.bpk");
        std::fs::create_dir_all(&root).expect("create temp dir");

        let device = Default::default();
        let config = LatentSeqMmFlowModelConfig::tiny_for_tests();
        let source = config.clone().init::<TestBackend>(&device);
        save_burnpack(&source, &bpk, "test TripoSplat flow").expect("save flow burnpack");

        let file_loaded =
            load_triposplat_flow_from_burnpack_file::<TestBackend>(&device, &bpk, &config)
                .expect("load flow burnpack");
        write_burnpack_parts_for_wasm(&bpk, 1, true).expect("write flow parts");
        std::fs::remove_file(&bpk).expect("remove base flow burnpack");
        let parts_loaded = load_model_from_burnpack_parts(
            &[bpk.clone()],
            "TripoSplat flow parity",
            false,
            || config.clone().init::<TestBackend>(&device),
            |model, bytes| {
                apply_triposplat_flow_burnpack_part_bytes(model, bytes)
                    .map_err(|err| err.to_string())
            },
        )
        .expect("load flow parts")
        .expect("parts model");

        let state = FlowState::<TestBackend>::deterministic_standard_normal(
            &device,
            1,
            config.q_token_length,
            config.in_channels,
            config.cam_channels,
            1234,
        );
        let cond = TripoSplatCondition {
            feature1: Tensor::zeros([1, 6, config.cond_channels], &device),
            feature2: Some(Tensor::zeros(
                [1, 4, config.cond2_channels.expect("cond2")],
                &device,
            )),
        };
        let t = Tensor::<TestBackend, 1>::from_floats([500.0], &device);
        let file_out = file_loaded.forward(state.clone(), t.clone(), cond.clone());
        let parts_out = parts_loaded.forward(state, t, cond);

        assert_close(
            &tensor_vec(file_out.latent),
            &tensor_vec(parts_out.latent),
            "flow latent",
        );
        assert_close(
            &tensor_vec(file_out.camera.expect("file camera")),
            &tensor_vec(parts_out.camera.expect("parts camera")),
            "flow camera",
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn flux2_single_burnpack_and_parts_loads_are_numerically_equivalent() {
        let root = unique_temp_dir("flux2_parts_parity");
        let bpk = root.join("flux2_vae_encoder.bpk");
        std::fs::create_dir_all(&root).expect("create temp dir");

        let device = Default::default();
        let config = burn_flux::Flux2VaeEncoderConfig::flux2();
        let source = config.clone().init::<TestBackend>(&device);
        save_burnpack(&source, &bpk, "test Flux2 VAE encoder").expect("save Flux2 VAE burnpack");

        let file_loaded = burn_flux::flux2_import::load_flux2_vae_encoder_from_burnpack_file::<
            TestBackend,
        >(&device, &bpk, &config)
        .expect("load Flux2 VAE burnpack");
        write_burnpack_parts_for_wasm(&bpk, 1, true).expect("write Flux2 VAE parts");
        std::fs::remove_file(&bpk).expect("remove base Flux2 VAE burnpack");
        let parts_loaded = load_model_from_burnpack_parts(
            &[bpk.clone()],
            "TripoSplat Flux2 VAE parity",
            false,
            || config.clone().init::<TestBackend>(&device),
            |model, bytes| {
                burn_flux::flux2_import::apply_flux2_vae_encoder_burnpack_part_bytes(model, bytes)
                    .map_err(|err| err.to_string())
            },
        )
        .expect("load Flux2 VAE parts")
        .expect("parts model");

        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let file_out = file_loaded.encode(input.clone(), true);
        let parts_out = parts_loaded.encode(input, true);

        assert_close(
            &tensor_vec(file_out),
            &tensor_vec(parts_out),
            "Flux2 VAE conditioning tokens",
        );

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn decoder_single_burnpack_and_parts_loads_are_numerically_equivalent() {
        let root = unique_temp_dir("decoder_parts_parity");
        let bpk = root.join("triposplat_vae_decoder.bpk");
        std::fs::create_dir_all(&root).expect("create temp dir");

        let device = Default::default();
        let octree_config = OctreeProbabilityFixedlenDecoderConfig::tiny_for_tests();
        let gs_config = ElasticGaussianFixedlenDecoderConfig::tiny_for_tests();
        let source = OctreeGaussianDecoder::<TestBackend>::new(
            &device,
            octree_config.clone(),
            gs_config.clone(),
        );
        save_burnpack(&source, &bpk, "test TripoSplat decoder").expect("save decoder burnpack");

        let file_loaded = load_triposplat_decoder_from_burnpack_file::<TestBackend>(
            &device,
            &bpk,
            &octree_config,
            &gs_config,
        )
        .expect("load decoder burnpack");
        write_burnpack_parts_for_wasm(&bpk, 1, true).expect("write decoder parts");
        std::fs::remove_file(&bpk).expect("remove base decoder burnpack");
        let parts_loaded = load_model_from_burnpack_parts(
            &[bpk.clone()],
            "TripoSplat decoder parity",
            false,
            || OctreeGaussianDecoder::new(&device, octree_config.clone(), gs_config.clone()),
            |model, bytes| {
                apply_triposplat_decoder_burnpack_part_bytes(model, bytes)
                    .map_err(|err| err.to_string())
            },
        )
        .expect("load decoder parts")
        .expect("parts model");

        let cond = Tensor::<TestBackend, 3>::zeros([1, 6, octree_config.cond_channels], &device);
        let sample =
            OctreeProbabilityFixedlenDecoder::<TestBackend>::sample_regular(&device, 1, 16, 2);
        let level = Tensor::<TestBackend, 1>::from_floats([4.0], &device);
        let total_points = Tensor::<TestBackend, 1>::from_floats([16.0], &device);

        let file_octree = file_loaded.octree.forward(
            sample.points.clone(),
            level.clone(),
            cond.clone(),
            Some(total_points.clone()),
        );
        let parts_octree = parts_loaded.octree.forward(
            sample.points.clone(),
            level,
            cond.clone(),
            Some(total_points),
        );
        assert_close(
            &tensor_vec(file_octree.logits),
            &tensor_vec(parts_octree.logits),
            "decoder octree logits",
        );

        let file_features = file_loaded.gs.forward(&sample, cond.clone());
        let parts_features = parts_loaded.gs.forward(&sample, cond);
        assert_close(
            &tensor_vec(file_features),
            &tensor_vec(parts_features),
            "decoder gaussian features",
        );

        let _ = std::fs::remove_dir_all(root);
    }
}

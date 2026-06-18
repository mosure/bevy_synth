use std::path::PathBuf;

use crate::artifact::{TripoSplatArtifactSet, TripoSplatBurnpackPrecision};
use crate::config::{TripoSplatOptions, normalize_num_gaussians};
use crate::decoder::TripoSplatDecodeReadbackStats;
use crate::gaussian::GaussianSplatCloud;
use crate::paths::resolve_triposplat_weights_root;

#[derive(Clone, Debug)]
pub struct TripoSplatPipelineConfig {
    pub weights_root: PathBuf,
    pub precision: TripoSplatBurnpackPrecision,
}

impl TripoSplatPipelineConfig {
    pub fn resolve(
        weights_root: Option<PathBuf>,
        precision: TripoSplatBurnpackPrecision,
    ) -> Result<Self, String> {
        Ok(Self {
            weights_root: resolve_triposplat_weights_root(weights_root.as_deref())?,
            precision,
        })
    }
}

#[derive(Debug)]
pub struct TripoSplatPipeline {
    config: TripoSplatPipelineConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripoSplatStageState {
    Ready,
    Scaffolded,
    Blocked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TripoSplatStageStatus {
    pub stage: &'static str,
    pub state: TripoSplatStageState,
    pub detail: &'static str,
}

pub const TRIPOSPLAT_STAGE_STATUS: [TripoSplatStageStatus; 9] = [
    TripoSplatStageStatus {
        stage: "artifacts",
        state: TripoSplatStageState::Ready,
        detail: "single-file .bpk and sharded .bpk.parts.json layouts are recognized for runtime-required TripoSplat components",
    },
    TripoSplatStageStatus {
        stage: "dinov3",
        state: TripoSplatStageState::Ready,
        detail: "DINOv3 ViT-H/16+ module shape, prefix token contract, and BurnPack import/load APIs are implemented",
    },
    TripoSplatStageStatus {
        stage: "flux2_vae_encoder",
        state: TripoSplatStageState::Ready,
        detail: "Flux2 VAE encoder feature2 conditioning contract and BurnPack import/load APIs are implemented",
    },
    TripoSplatStageStatus {
        stage: "flow",
        state: TripoSplatStageState::Scaffolded,
        detail: "latent sequence flow module, seeded noise init, Euler CFG sampler, and BurnPack import/load APIs compile; exact upstream parity is not complete",
    },
    TripoSplatStageStatus {
        stage: "octree_decoder",
        state: TripoSplatStageState::Scaffolded,
        detail: "octree probability module, decoder BurnPack load path, and host systematic traversal compile; GPU-resident sampling parity is not complete",
    },
    TripoSplatStageStatus {
        stage: "gaussian_decoder",
        state: TripoSplatStageState::Scaffolded,
        detail: "elastic Gaussian decoder module and decoder BurnPack load path compile; production GPU-resident splat materialization is not complete",
    },
    TripoSplatStageStatus {
        stage: "runtime_loader",
        state: TripoSplatStageState::Ready,
        detail: "component import/load APIs, single-file vs parts loader parity, preprocessed-tensor runtime inference, upstream-style multi-density latent replay, burn_synth native asset wiring, and bevy_synth_runtime asset transport are implemented",
    },
    TripoSplatStageStatus {
        stage: "image_reference_parity",
        state: TripoSplatStageState::Blocked,
        detail: "direct TripoSplatPipeline image-file entrypoint and upstream-weight stage/reference parity thresholds are not complete",
    },
    TripoSplatStageStatus {
        stage: "wrapper_rendering",
        state: TripoSplatStageState::Blocked,
        detail: "native Bevy can transport/export Gaussian-splat assets and spawn a bounded mesh preview; wasm can load sharded TripoSplat components and return splat/PLY assets; production splat rendering is not complete",
    },
];

#[derive(Clone, Debug)]
pub struct TripoSplatRunOutput {
    pub splats: GaussianSplatCloud,
    pub options: TripoSplatOptions,
    pub decode_readbacks: TripoSplatDecodeReadbackStats,
}

#[derive(Clone, Debug)]
pub struct TripoSplatMultiRunOutput {
    pub splats: Vec<GaussianSplatCloud>,
    pub options: Vec<TripoSplatOptions>,
    pub decode_readbacks: TripoSplatDecodeReadbackStats,
}

impl TripoSplatPipeline {
    pub fn new(config: TripoSplatPipelineConfig) -> Result<Self, String> {
        TripoSplatArtifactSet::new(&config.weights_root, config.precision).validate_burnpacks()?;
        Ok(Self { config })
    }

    pub fn from_pretrained(
        weights_root: Option<PathBuf>,
        precision: TripoSplatBurnpackPrecision,
    ) -> Result<Self, String> {
        Self::new(TripoSplatPipelineConfig::resolve(weights_root, precision)?)
    }

    pub fn config(&self) -> &TripoSplatPipelineConfig {
        &self.config
    }

    pub fn stage_status(&self) -> &'static [TripoSplatStageStatus] {
        &TRIPOSPLAT_STAGE_STATUS
    }

    #[cfg(feature = "import")]
    pub fn load_runtime_components<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
    ) -> Result<crate::TripoSplatRuntimeComponents<B>, String> {
        let artifacts =
            TripoSplatArtifactSet::new(&self.config.weights_root, self.config.precision);
        crate::import::load_triposplat_runtime_components(device, &artifacts)
            .map_err(|err| err.to_string())
    }

    #[cfg(feature = "import")]
    pub fn load_runtime_components_with_compute_dtypes<B: burn::prelude::Backend>(
        &self,
        device: &B::Device,
        compute_dtypes: crate::import::TripoSplatRuntimeComputeDtypes,
    ) -> Result<crate::TripoSplatRuntimeComponents<B>, String> {
        let artifacts =
            TripoSplatArtifactSet::new(&self.config.weights_root, self.config.precision);
        crate::import::load_triposplat_runtime_components_with_compute_dtypes_and_callback(
            device,
            &artifacts,
            compute_dtypes,
            |_| Ok::<(), Box<dyn std::error::Error>>(()),
        )
        .map_err(|err| err.to_string())
    }

    pub fn infer_image(
        &mut self,
        _image_path: impl AsRef<std::path::Path>,
        options: TripoSplatOptions,
    ) -> Result<TripoSplatRunOutput, String> {
        let num_gaussians = normalize_num_gaussians(options.num_gaussians)?;
        let blocked = self
            .stage_status()
            .iter()
            .filter(|stage| stage.state == TripoSplatStageState::Blocked)
            .map(|stage| format!("{} ({})", stage.stage, stage.detail))
            .collect::<Vec<_>>()
            .join("; ");
        Err(format!(
            "TripoSplat image-file inference is incomplete; artifacts are present at {} and neural modules can run from preprocessed tensors, but the file preprocessing/runtime wrapper path is blocked before execution. num_gaussians={} blocked_stages={}",
            self.config.weights_root.display(),
            num_gaussians,
            blocked
        ))
    }

    pub fn debug_output(options: TripoSplatOptions) -> TripoSplatRunOutput {
        TripoSplatRunOutput {
            splats: GaussianSplatCloud::canonical_debug_cloud(),
            options,
            decode_readbacks: TripoSplatDecodeReadbackStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::TRIPOSPLAT_ARTIFACTS;

    #[test]
    fn stage_status_marks_compiled_modules_and_runtime_blockers() {
        assert!(
            TRIPOSPLAT_STAGE_STATUS
                .iter()
                .any(|stage| stage.stage == "dinov3" && stage.state == TripoSplatStageState::Ready)
        );
        assert!(
            TRIPOSPLAT_STAGE_STATUS
                .iter()
                .any(|stage| stage.stage == "flux2_vae_encoder"
                    && stage.state == TripoSplatStageState::Ready)
        );
        assert!(
            TRIPOSPLAT_STAGE_STATUS
                .iter()
                .any(|stage| stage.stage == "runtime_loader"
                    && stage.state == TripoSplatStageState::Ready)
        );
        assert!(TRIPOSPLAT_STAGE_STATUS.iter().any(|stage| {
            stage.stage == "image_reference_parity" && stage.state == TripoSplatStageState::Blocked
        }));
    }

    #[test]
    fn infer_error_names_loader_blocker_not_missing_modules() {
        let root = std::env::temp_dir().join(format!(
            "burn_triposplat_pipeline_parts_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        for artifact in TRIPOSPLAT_ARTIFACTS {
            let manifest = artifact.parts_manifest_path(&root, TripoSplatBurnpackPrecision::F16);
            std::fs::create_dir_all(manifest.parent().expect("manifest parent"))
                .expect("create artifact dir");
            std::fs::write(manifest, "{}").expect("write parts manifest");
        }
        let config = TripoSplatPipelineConfig {
            weights_root: root.clone(),
            precision: TripoSplatBurnpackPrecision::F16,
        };
        let mut pipeline = TripoSplatPipeline::new(config).expect("parts manifests accepted");
        let err = pipeline
            .infer_image("input.png", TripoSplatOptions::default())
            .expect_err("full runtime loader remains blocked");
        assert!(err.contains("image-file inference is incomplete"));
        assert!(err.contains("image_reference_parity"));
        assert!(!err.contains("modules still need to be ported"));

        let _ = std::fs::remove_dir_all(root);
    }
}
